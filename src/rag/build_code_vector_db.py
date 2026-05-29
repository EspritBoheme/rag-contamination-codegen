"""
构建编程代码向量库
- 从 GitHub clone 高质量仓库
- 分层 chunk: repo -> module -> file -> function
- 存入 ChromaDB，支持分层检索
- 目标 ~5GB
"""
import os
import ast
import json
import re
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# ============================================================
# Config
# ============================================================
VECTOR_DB_DIR = "./vector_db"
CACHE_DIR = "./.repo_cache"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 目标仓库列表 (owner/repo, timeout_seconds)
REPOS = [
    # Python: AI/ML
    ("huggingface/transformers", 600),
    # Python: Web
    ("fastapi/fastapi", 120),
    ("django/django", 300),
    ("pallets/flask", 60),
    # Python: Tools
    ("psf/requests", 60),
    ("sqlalchemy/sqlalchemy", 120),
    ("redis/redis", 120),
    # Go
    ("golang/go", 900),
    ("gin-gonic/gin", 90),
    # Rust
    ("tokio-rs/tokio", 180),
    ("serde-rs/serde", 60),
    # JS/TS
    ("expressjs/express", 90),
    ("lodash/lodash", 60),
    ("nestjs/nest", 120),
    # Java
    ("google/gson", 60),
    ("junit-team/junit5", 120),
    # C/C++
    ("nlohmann/json", 60),
    ("libuv/libuv", 60),
]

LANGUAGE_MAP = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

# ============================================================
# Step 1: Shallow clone repos
# ============================================================
def clone_repo(repo_name: str, target_dir: Path, timeout: int = 300, max_retries: int = 2) -> bool:
    """Shallow clone single branch. Retry with backoff on failure."""
    if (target_dir / ".git").exists():
        print(f"  [SKIP] {repo_name} already exists")
        return True
    for attempt in range(max_retries + 1):
        url = f"https://github.com/{repo_name}.git"
        print(f"  Cloning {repo_name} (timeout={timeout}s, attempt {attempt+1}/{max_retries+1})...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target_dir)],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                print(f"  [OK] {repo_name}")
                return True
            print(f"  [FAIL] {repo_name}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {repo_name} after {timeout}s")
        # Clean partial clone before retry
        if target_dir.exists():
            shutil.rmtree(str(target_dir), ignore_errors=True)
        if attempt < max_retries:
            wait = 2 ** attempt * 5
            print(f"  Retry in {wait}s...")
            time.sleep(wait)
    return False


def clone_all_repos():
    repos_dir = Path(CACHE_DIR) / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for repo, timeout in REPOS:
        safe_name = repo.replace("/", "__")
        target = repos_dir / safe_name
        ok = clone_repo(repo, target, timeout)
        results[repo] = {"path": str(target), "ok": ok}
    return results


# ============================================================
# Step 2: Hierarchical chunking
# ============================================================
def extract_functions_py(source: str, file_path: str) -> List[Dict]:
    """Extract function-level chunks from Python file."""
    chunks = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_text = ast.get_source_segment(source, node)
                if func_text and len(func_text) > 30:
                    chunks.append({
                        "type": "function",
                        "name": node.name,
                        "content": func_text,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                    })
            elif isinstance(node, ast.ClassDef):
                class_text = ast.get_source_segment(source, node)
                if class_text:
                    chunks.append({
                        "type": "class",
                        "name": node.name,
                        "content": class_text,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                    })
    except SyntaxError:
        pass
    return chunks


def extract_functions_generic(source: str, ext: str) -> List[Dict]:
    """Function extraction via regex for non-Python languages.
    Note: regex can't match nested braces — function bodies with inner
    braces will be truncated. Good enough for embedding, not for execution."""
    chunks = []
    patterns = {
        ".go": [
            r'func\s+(?:\([^)]*\)\s+)?\w+[^{]*\{[^}]*\}',
        ],
        ".js": [
            r'(?:async\s+)?function(?:\s+\w+)?\s*\([^)]*\)[^{]*\{[^}]*\}',
            r'\w+\s*=\s*(?:async\s+)?function\s*\([^)]*\)[^{]*\{[^}]*\}',
        ],
        ".ts": [
            r'(?:async\s+)?function(?:\s+\w+)?\s*\([^)]*\)\s*(?::\s*\w+)?[^{]*\{[^}]*\}',
            r'\w+\s*=\s*(?:async\s+)?function\s*\([^)]*\)\s*(?::\s*\w+)?[^{]*\{[^}]*\}',
            r'\w+\s*:\s*(?:async\s+)?\([^)]*\)\s*=>[^{]*\{[^}]*\}',
        ],
        ".rs": [
            r'(?:pub\s+)?(?:unsafe\s+)?fn\s+\w+[^{;]*\{[^}]*\}',
        ],
        ".java": [
            r'(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:\w+(?:<[^>]*>)?)\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{[^}]*\}',
        ],
        ".c": [
            r'(?:\w+\s+)+\w+\s*\([^)]*\)\s*\{[^}]*\}',
        ],
        ".cpp": [
            r'(?:\w+(?:::))?\w+\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\{[^}]*\}',
            r'(?:\w+\s+)+\w+\s*\([^)]*\)\s*(?:const\s*)?\{[^}]*\}',
        ],
    }
    pats = patterns.get(ext)
    if not pats:
        return chunks
    for pat in pats:
        for match in re.finditer(pat, source, re.DOTALL):
            text = match.group().strip()
            if len(text) < 30:
                continue
            # Extract function name for better labeling
            name = f"match_{len(chunks)}"
            name_match = re.search(r'(?:fn|func|function)\s+(\w+)', text)
            if name_match:
                name = name_match.group(1)
            chunks.append({
                "type": "function",
                "name": name,
                "content": text,
                "start_line": source[:match.start()].count("\n") + 1,
                "end_line": source[:match.end()].count("\n") + 1,
            })
    return chunks


def chunk_file(file_path: Path, repo_name: str, module_path: str) -> List[Dict]:
    """Chunk single file into hierarchical levels."""
    ext = file_path.suffix.lower()
    lang = LANGUAGE_MAP.get(ext)
    if not lang:
        return []

    # Skip generated/test files
    if any(p in file_path.name for p in ["__pycache__", "node_modules", ".git"]):
        return []
    if "test" in file_path.name.lower() and ext != ".py":
        return []
    if file_path.name in ("setup.py", "version.py", "__init__.py"):
        pass  # Keep these

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rel_path = file_path.as_posix()
    chunks = []

    # Level 3: File-level chunk (whole file summary)
    docstring = ""
    if ext == ".py":
        try:
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree) or ""
        except SyntaxError:
            pass

    file_summary = f"# {rel_path}\n\n"
    if docstring:
        file_summary += docstring + "\n\n"
    # First 50 lines as summary
    first_lines = source.split("\n")[:50]
    file_summary += "\n".join(first_lines)

    chunks.append({
        "level": 3,  # file level
        "type": "file",
        "repo": repo_name,
        "module": module_path,
        "file_path": rel_path,
        "language": lang,
        "name": rel_path,
        "content": file_summary,
        "docstring": docstring,
    })

    # Level 4: Function/class chunks
    if ext == ".py":
        func_chunks = extract_functions_py(source, rel_path)
    else:
        func_chunks = extract_functions_generic(source, ext)

    for fc in func_chunks:
        chunks.append({
            "level": 4,
            "type": fc["type"],
            "repo": repo_name,
            "module": module_path,
            "file_path": rel_path,
            "language": lang,
            "name": fc["name"],
            "content": fc["content"],
            "start_line": fc["start_line"],
            "end_line": fc["end_line"],
            "docstring": "",
        })

    return chunks


def chunk_repo(repo_name: str, repo_path: str) -> List[Dict]:
    """Chunk entire repo hierarchically."""
    all_chunks = []
    repo_path = Path(repo_path)
    repo_short = repo_name.split("/")[-1]

    # Level 1: Repo-level summary
    readme = ""
    for readme_name in ["README.md", "README.rst", "README"]:
        rp = repo_path / readme_name
        if rp.exists():
            try:
                readme = rp.read_text(encoding="utf-8", errors="ignore")[:2000]
            except Exception:
                pass
            break

    all_chunks.append({
        "level": 1,
        "type": "repo",
        "repo": repo_name,
        "module": "",
        "file_path": "",
        "language": "",
        "name": repo_name,
        "content": f"Repository: {repo_name}\n\n{readme}" if readme else repo_name,
        "docstring": "",
    })

    # Level 2: Module-level (top-level dirs)
    for item in sorted(repo_path.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            if item.name in ("__pycache__", "node_modules", "test", "tests", "venv", "env"):
                continue
            all_chunks.append({
                "level": 2,
                "type": "module",
                "repo": repo_name,
                "module": item.name,
                "file_path": item.as_posix(),
                "language": "",
                "name": f"{repo_short}/{item.name}",
                "content": f"Module: {repo_short}/{item.name}",
                "docstring": "",
            })

    # Level 3+4: Walk files
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden/node_modules
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("node_modules", "__pycache__", "venv", "env", ".git")]
        for fname in files:
            fpath = Path(root) / fname
            rel_dir = Path(root).relative_to(repo_path).as_posix()
            module_path = rel_dir.split("/")[0] if rel_dir != "." else ""
            chunks = chunk_file(fpath, repo_name, module_path)
            all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# Step 3: Estimate embedding cost
# ============================================================
def estimate_chunks(all_chunks: List[Dict]):
    """Print stats."""
    by_level = {}
    by_lang = {}
    total_chars = 0
    for c in all_chunks:
        level = c["level"]
        by_level[level] = by_level.get(level, 0) + 1
        if c.get("language"):
            by_lang[c["language"]] = by_lang.get(c["language"], 0) + 1
        total_chars += len(c["content"])

    print(f"\n=== Chunk Stats ===")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    for lv in sorted(by_level):
        print(f"  Level {lv}: {by_level[lv]} chunks")
    print(f"  By language: {json.dumps(by_lang, ensure_ascii=False)}")


# ============================================================
# Step 4: Embed & store in ChromaDB
# ============================================================
def build_vector_db(all_chunks: List[Dict]):
    """
    Build ChromaDB with hierarchical collections.
    Uses ChromaDB built-in ONNX embedding (all-MiniLM-L6-v2) — no torch needed.
    """
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    print("\nLoading ChromaDB with ONNX embedding (all-MiniLM-L6-v2)...")
    ef = ONNXMiniLM_L6_V2()
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # 3 collections for hierarchical retrieval with tuned HNSW params
    # M=32: more connections → better recall. ef_construction=200: better index quality
    hnsw_cfg = {"hnsw:space": "cosine", "hnsw:construction_metric": "cosine",
                 "hnsw:M": 32, "hnsw:ef_construction": 200}

    summary_col = chroma_client.get_or_create_collection(
        "summary", metadata={**hnsw_cfg, "description": "Level 1-2: repo + module"},
        embedding_function=ef,
    )
    chunks_col = chroma_client.get_or_create_collection(
        "chunks", metadata={**hnsw_cfg, "description": "Level 3: file-level"},
        embedding_function=ef,
    )
    details_col = chroma_client.get_or_create_collection(
        "details", metadata={**hnsw_cfg, "description": "Level 4: function/class"},
        embedding_function=ef,
    )

    # Batch insert
    batch_size = 128
    summary_batch = {"ids": [], "documents": [], "metadatas": []}
    chunks_batch = {"ids": [], "documents": [], "metadatas": []}
    details_batch = {"ids": [], "documents": [], "metadatas": []}

    # Dedup by content hash
    seen_hashes = set()

    for i, chunk in enumerate(all_chunks):
        content_hash = hash(chunk["content"][:200])
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        chunk_id = f"{chunk['level']}_{i}"
        meta = {
            "level": chunk["level"],
            "type": chunk["type"],
            "repo": chunk["repo"],
            "module": chunk["module"],
            "file_path": chunk["file_path"],
            "language": chunk["language"],
            "name": chunk["name"],
        }

        if chunk["level"] <= 2:
            summary_batch["ids"].append(chunk_id)
            summary_batch["documents"].append(chunk["content"])
            summary_batch["metadatas"].append(meta)
        elif chunk["level"] == 3:
            chunks_batch["ids"].append(chunk_id)
            chunks_batch["documents"].append(chunk["content"])
            chunks_batch["metadatas"].append(meta)
        else:
            details_batch["ids"].append(chunk_id)
            details_batch["documents"].append(chunk["content"])
            details_batch["metadatas"].append(meta)

        # Batch insert
        for batch, col in [
            (summary_batch, summary_col),
            (chunks_batch, chunks_col),
            (details_batch, details_col),
        ]:
            if len(batch["ids"]) >= batch_size:
                col.add(ids=batch["ids"], documents=batch["documents"],
                        metadatas=batch["metadatas"])
                batch["ids"], batch["documents"], batch["metadatas"] = [], [], []

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(all_chunks)} chunks...")

    # Flush remaining
    for batch, col in [
        (summary_batch, summary_col),
        (chunks_batch, chunks_col),
        (details_batch, details_col),
    ]:
        if batch["ids"]:
            col.add(ids=batch["ids"], documents=batch["documents"],
                    metadatas=batch["metadatas"])

    print(f"\n=== DB Built ===")
    print(f"  summary_col: {summary_col.count()}")
    print(f"  chunks_col: {chunks_col.count()}")
    print(f"  details_col: {details_col.count()}")
    print(f"  DB path: {VECTOR_DB_DIR}")


# ============================================================
# Hierarchical Search
# ============================================================
class CodeRetriever:
    """Hierarchical code retriever using ONNX embedding."""

    def __init__(self, db_path: str = VECTOR_DB_DIR):
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = ONNXMiniLM_L6_V2()
        self.summary_col = self.client.get_collection("summary", embedding_function=self.ef)
        self.chunks_col = self.client.get_collection("chunks", embedding_function=self.ef)
        self.details_col = self.client.get_collection("details", embedding_function=self.ef)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.7):
        """Three-level hierarchical search."""
        # Level 1: Summary
        summary_results = self.summary_col.query(query_texts=[query], n_results=top_k)

        # Build repo/module filter from summary hits
        repo_filter = None
        module_filter = None
        if summary_results["metadatas"][0]:
            for meta in summary_results["metadatas"][0]:
                if meta.get("repo") and not repo_filter:
                    repo_filter = meta["repo"]
                if meta.get("module") and meta["module"]:
                    module_filter = meta["module"]

        # Level 2: File chunks (optionally filtered)
        where = {}
        if repo_filter:
            where["repo"] = repo_filter

        chunk_results = self.chunks_col.query(
            query_texts=[query], n_results=top_k, where=where
        )

        # Level 3: Details (fallback if confidence low)
        max_dist = max(chunk_results["distances"][0]) if chunk_results["distances"][0] else 1.0
        if max_dist > threshold:
            detail_results = self.details_col.query(
                query_texts=[query], n_results=top_k,
                where={"repo": repo_filter} if repo_filter else None
            )
        else:
            detail_results = None

        return {
            "summary": summary_results,
            "chunks": chunk_results,
            "details": detail_results,
        }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build code vector DB")
    parser.add_argument("--clone-only", action="store_true", help="Only clone repos")
    parser.add_argument("--chunk-only", action="store_true", help="Only chunk & build DB")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing DB and rebuild")
    parser.add_argument("--validate", action="store_true", help="Check DB integrity")
    parser.add_argument("--search", type=str, default=None,
                        help="Test search query")
    args = parser.parse_args()

    # --rebuild: wipe existing DB before fresh start
    if args.rebuild and os.path.isdir(VECTOR_DB_DIR):
        print("=== [REBUILD] Deleting existing vector_db ===")
        for f in os.listdir(VECTOR_DB_DIR):
            fp = os.path.join(VECTOR_DB_DIR, f)
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            else:
                os.remove(fp)
        print("  Done.")

    # --validate: check DB integrity without modifying
    if args.validate:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            ef = ONNXMiniLM_L6_V2()
            client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            for name in ["summary", "chunks", "details"]:
                col = client.get_collection(name, embedding_function=ef)
                cnt = col.count()
                print(f"  [OK] {name}: {cnt} docs")
            print("  DB integrity check passed.")
        except Exception as e:
            print(f"  [FAIL] DB error: {e}")
        exit(0)

    if not args.chunk_only:
        print("=== Step 1: Clone repos ===")
        repos = clone_all_repos()
    else:
        # Load existing — try both naming conventions
        repos_dir = Path(CACHE_DIR) / "repos"
        repos = {}
        for repo_name, _ in REPOS:
            safe_name = repo_name.replace("/", "__")
            target = repos_dir / safe_name
            # Try __ format first, then bare name
            if not target.exists():
                bare_name = repo_name.split("/")[-1]
                target = repos_dir / bare_name
            if target.exists():
                repos[repo_name] = {"path": str(target), "ok": True}

    if args.clone_only:
        print("\nClone complete. Run without --clone-only to build DB.")
        exit(0)

    if args.search:
        retriever = CodeRetriever()
        result = retriever.search(args.search)
        print("\n=== Summary hits ===")
        for i, (doc, meta) in enumerate(zip(
                result["summary"]["documents"][0],
                result["summary"]["metadatas"][0])):
            print(f"  [{i}] {meta.get('repo','?')} | {meta.get('name','')} | {doc[:100]}...")
        print("\n=== File chunk hits ===")
        for i, (doc, meta) in enumerate(zip(
                result["chunks"]["documents"][0],
                result["chunks"]["metadatas"][0])):
            print(f"  [{i}] {meta.get('repo','?')} | {meta.get('file_path','')}")
        print("\n=== Detail hits ===")
        if result["details"]:
            for i, (doc, meta) in enumerate(zip(
                    result["details"]["documents"][0],
                    result["details"]["metadatas"][0])):
                print(f"  [{i}] {meta.get('name','')} | {doc[:100]}...")
        exit(0)

    print("\n=== Step 2: Chunk repos ===")
    all_chunks = []
    for repo_name, info in repos.items():
        if not info["ok"]:
            continue
        print(f"  Chunking {repo_name}...")
        chunks = chunk_repo(repo_name, info["path"])
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks")

    # Estimate size
    estimate_chunks(all_chunks)

    print(f"\n=== Step 3: Build vector DB ===")
    build_vector_db(all_chunks)

    print(f"\nDone! Vector DB at {VECTOR_DB_DIR}/")
    print(f"\nTest search: python build_code_vector_db.py --search \"python async await pattern\"")
