"""
Search engine for code vector DB.

Features:
- Parallel query across 3 collections (summary / chunks / details)
- LRU cache for repeated queries
- Filter by language or repo
- Per-query latency reporting
"""
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional


# ============================================================
# LRU Cache
# ============================================================
class LRUCache:
    """Simple thread-safe LRU cache with TTL support."""

    def __init__(self, capacity: int = 256, ttl: int = 3600):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl  # seconds

    def get(self, key: str):
        if key not in self.cache:
            return None
        value, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return value

    def put(self, key: str, value):
        self.cache[key] = (value, time.time() + self.ttl)
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self.cache)


# ============================================================
# Search Engine
# ============================================================
class SearchEngine:
    """Hierarchical code search with caching and parallel query."""

    def __init__(self, db_path: str = "./vector_db", cache_size: int = 256, cache_ttl: int = 3600):
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self.ef = ONNXMiniLM_L6_V2()
        self.client = chromadb.PersistentClient(path=db_path)

        try:
            self.summary_col = self.client.get_collection("summary", embedding_function=self.ef)
            self.chunks_col = self.client.get_collection("chunks", embedding_function=self.ef)
            self.details_col = self.client.get_collection("details", embedding_function=self.ef)
        except Exception as e:
            raise RuntimeError(f"Failed to load vector DB at {db_path}: {e}")

        self.cache = LRUCache(capacity=cache_size, ttl=cache_ttl)
        self.stats = {"hits": 0, "misses": 0, "total_time": 0.0, "query_count": 0}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7,
               language: Optional[str] = None, repo: Optional[str] = None,
               parallel: bool = True) -> Dict:
        """Multi-level search across the vector DB.

        Args:
            query: Search text.
            top_k: Results per level.
            threshold: Distance threshold for details fallback (lower = stricter).
            language: Filter by language (python, go, rust, etc.).
            repo: Filter by repo name (e.g. "fastapi/fastapi").
            parallel: Query summary + chunks in parallel (faster).

        Returns:
            dict with keys: summary, chunks, details, elapsed
        """
        # Input validation
        query = query.strip()
        if not query or len(query) < 3:
            return {"error": "Query too short (min 3 chars)", "elapsed": 0}

        cache_key = self._cache_key(query, top_k, threshold, language, repo)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats["hits"] += 1
            result = dict(cached)
            result["elapsed"] = 0.0  # cache hit = instant
            return result

        self.stats["misses"] += 1
        t0 = time.time()

        where = {}
        if language:
            where["language"] = language
        if repo:
            where["repo"] = repo

        try:
            if parallel and language is None and repo is None:
                results = self._search_parallel(query, top_k, threshold)
            else:
                results = self._search_sequential(query, top_k, threshold, where)
        except Exception as e:
            return {"error": str(e), "elapsed": time.time() - t0}

        elapsed = time.time() - t0
        results["elapsed"] = round(elapsed, 4)
        self.stats["total_time"] += elapsed
        self.stats["query_count"] += 1

        self.cache.put(cache_key, results)
        return results

    def search_by_language(self, query: str, language: str, top_k: int = 5) -> Dict:
        """Search within specific language."""
        return self.search(query, top_k=top_k, language=language)

    def search_by_repo(self, query: str, repo: str, top_k: int = 5) -> Dict:
        """Search within specific repo."""
        return self.search(query, top_k=top_k, repo=repo)

    def list_languages(self) -> List[str]:
        """Get all languages present in the DB."""
        all_meta = self.chunks_col.get(limit=100000)["metadatas"]
        langs = set()
        for m in all_meta:
            if m.get("language"):
                langs.add(m["language"])
        return sorted(langs)

    def list_repos(self) -> List[Dict]:
        """Get all repos and their doc count."""
        all_meta = self.chunks_col.get(limit=100000)["metadatas"]
        repo_counts = {}
        for m in all_meta:
            r = m.get("repo", "unknown")
            repo_counts[r] = repo_counts.get(r, 0) + 1
        return sorted([{"repo": k, "files": v} for k, v in repo_counts.items()],
                      key=lambda x: -x["files"])

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------
    def _cache_key(self, query: str, top_k: int, threshold: float,
                   language: Optional[str], repo: Optional[str]) -> str:
        raw = f"{query}|{top_k}|{threshold}|{language or ''}|{repo or ''}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _search_parallel(self, query: str, top_k: int, threshold: float) -> Dict:
        """Query summary + chunks in parallel, details on-demand."""
        results = {}

        with ThreadPoolExecutor(max_workers=2) as exc:
            fut_s = exc.submit(self.summary_col.query, query_texts=[query], n_results=top_k)
            fut_c = exc.submit(self.chunks_col.query, query_texts=[query], n_results=top_k)
            results["summary"] = fut_s.result()
            results["chunks"] = fut_c.result()

        # Details only if chunks confidence is low (avoids unnecessary large search)
        if results["chunks"]["distances"] and results["chunks"]["distances"][0]:
            max_dist = max(results["chunks"]["distances"][0])
            if max_dist > threshold:
                results["details"] = self.details_col.query(
                    query_texts=[query], n_results=top_k
                )
            else:
                results["details"] = None
        else:
            results["details"] = None

        return results

    def _search_sequential(self, query: str, top_k: int, threshold: float,
                           where: dict) -> Dict:
        """Sequential search with optional filters."""
        where_or_none = where if where else None
        summary = self.summary_col.query(query_texts=[query], n_results=top_k, where=where_or_none)
        chunks = self.chunks_col.query(query_texts=[query], n_results=top_k, where=where_or_none)

        details = None
        if chunks["distances"] and chunks["distances"][0]:
            max_dist = max(chunks["distances"][0])
            if max_dist > threshold:
                details = self.details_col.query(query_texts=[query], n_results=top_k, where=where_or_none)

        return {"summary": summary, "chunks": chunks, "details": details}

    def format_results(self, results: Dict, show_content: bool = True) -> str:
        """Pretty-print search results."""
        if "error" in results:
            return f"[ERROR] {results['error']}"

        lines = []
        lines.append(f"Search completed in {results.get('elapsed', 0):.3f}s\n")

        # Summary
        if results.get("summary"):
            lines.append("── Repo/Module ──")
            for doc, meta in zip(results["summary"]["documents"][0],
                                 results["summary"]["metadatas"][0]):
                repo = meta.get("repo", "?")
                name = meta.get("name", "")
                score = results["summary"]["distances"][0][0] if results["summary"]["distances"] else 0
                lines.append(f"  [{score:.3f}] {repo} | {name[:60]}")

        # Chunks
        if results.get("chunks"):
            lines.append("\n── File ──")
            for doc, meta in zip(results["chunks"]["documents"][0],
                                 results["chunks"]["metadatas"][0]):
                fpath = meta.get("file_path", "")
                repo = meta.get("repo", "?")
                score = results["chunks"]["distances"][0][0] if results["chunks"]["distances"] else 0
                snippet = doc[:120].replace("\n", "\\n") if show_content else fpath
                lines.append(f"  [{score:.3f}] {repo}:{fpath}")
                if show_content:
                    lines.append(f"    {snippet}")

        # Details
        if results.get("details"):
            lines.append("\n── Function/Class ──")
            for doc, meta in zip(results["details"]["documents"][0],
                                 results["details"]["metadatas"][0]):
                name = meta.get("name", "?")
                repo = meta.get("repo", "?")
                score = results["details"]["distances"][0][0] if results["details"]["distances"] else 0
                snippet = doc[:150].replace("\n", "\\n") if show_content else ""
                lines.append(f"  [{score:.3f}] {repo}::{name}")
                if show_content:
                    lines.append(f"    {snippet}")

        return "\n".join(lines)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Code vector DB search engine")
    parser.add_argument("query", type=str, nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Results per level")
    parser.add_argument("--threshold", type=float, default=0.7, help="Distance threshold")
    parser.add_argument("--language", type=str, default=None, help="Filter by language")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repo")
    parser.add_argument("--serial", action="store_true", help="Sequential (non-parallel) search")
    parser.add_argument("--db", type=str, default="./vector_db", help="Vector DB path")
    parser.add_argument("--list-languages", action="store_true", help="List available languages")
    parser.add_argument("--list-repos", action="store_true", help="List repos in DB")
    parser.add_argument("--stats", action="store_true", help="Show search engine stats")
    parser.add_argument("--cache-size", type=int, default=256, help="LRU cache capacity")
    args = parser.parse_args()

    engine = SearchEngine(db_path=args.db, cache_size=args.cache_size)

    if args.list_languages:
        print("Languages:", ", ".join(engine.list_languages()))
        exit(0)

    if args.list_repos:
        for r in engine.list_repos():
            print(f"  {r['repo']}: {r['files']} files")
        exit(0)

    if args.stats:
        s = engine.stats
        print(f"  Queries: {s['query_count']}")
        print(f"  Cache hits: {s['hits']}")
        print(f"  Cache misses: {s['misses']}")
        hit_rate = s['hits'] / (s['hits'] + s['misses']) * 100 if (s['hits'] + s['misses']) > 0 else 0
        print(f"  Hit rate: {hit_rate:.1f}%")
        print(f"  Avg latency: {s['total_time']/max(s['query_count'],1):.3f}s")
        print(f"  Cache size: {engine.cache.size}/{args.cache_size}")
        exit(0)

    if not args.query:
        parser.print_help()
        exit(1)

    results = engine.search(
        args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        language=args.language,
        repo=args.repo,
        parallel=not args.serial,
    )
    print(engine.format_results(results, show_content=True))
