"""
扩展现有 ChromaDB 向量库: 新增 Java/Kotlin/ArkTS/Python 仓库
- 不重复已索引的仓库
- 下载 ~1-2GB
- Phase 4 完成后执行
用法:
    python expand_vector_db_v2.py             # 完整流程
    python expand_vector_db_v2.py --clone-only  # 只下载
    python expand_vector_db_v2.py --chunk-only  # 跳过下载,用已有仓库
    python expand_vector_db_v2.py --dry-run     # 只显示计划
"""
import ast, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

VECTOR_DB_DIR = "./vector_db"
CACHE_DIR = "./.repo_cache"
NEW_REPOS_DIR = Path(CACHE_DIR) / "repos_v2"
NEW_REPOS_DIR.mkdir(parents=True, exist_ok=True)

# ==== 新仓库列表 (Java, Kotlin, ArkTS/TS, Python) ====
# 已索引的: transformers, fastapi, django, flask, requests, sqlalchemy (python)
#              gson, junit5 (java), express, lodash (js), nestjs (ts)
#              gin (go), tokio, serde (rust), nlohmann/json (cpp), libuv (c)
# 以下全是不重复的
NEW_REPOS: List[Tuple[str, int, str]] = [
    # ===== Python (新) =====
    ("pandas-dev/pandas", 300, "python"),
    ("scikit-learn/scikit-learn", 300, "python"),
    ("pytest-dev/pytest", 120, "python"),
    ("celery/celery", 120, "python"),
    ("apache/airflow", 300, "python"),
    ("apache/spark", 600, "python"),
    # ===== Java (新) =====
#    ("spring-projects/spring-boot", 300, "java"),  # too big, times out
    ("apache/kafka", 300, "java"),
    ("netty/netty", 180, "java"),
#    ("apache/dubbo", 180, "java"),  # network issues
    ("reactivex/rxjava", 120, "java"),
    ("square/okhttp", 120, "java"),
    # ===== Kotlin =====
#    ("JetBrains/kotlin", 600, "kotlin"),  # too big, times out
    ("Kotlin/kotlinx.coroutines", 120, "kotlin"),
    ("ktorio/ktor", 180, "kotlin"),
    ("JetBrains/compose-multiplatform", 300, "kotlin"),
    ("square/okio", 120, "kotlin"),
    # ===== ArkTS / TypeScript =====
    ("microsoft/TypeScript", 300, "typescript"),
    ("jestjs/jest", 180, "typescript"),
    ("prettier/prettier", 120, "typescript"),
    ("type-challenges/type-challenges", 60, "typescript"),
]

LANGUAGE_MAP = {
    ".py": "python", ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
}

# Phase 3 repos for details rebuild
PHASE3_REPOS: List[Tuple[str, int, str]] = [
    ("huggingface/transformers", 180, "python"),
    ("pallets/flask", 60, "python"),
    ("psf/requests", 60, "python"),
    ("redis/redis", 60, "python"),
    ("sqlalchemy/sqlalchemy", 60, "python"),
]


def get_local_phase3_paths() -> Dict[str, str]:
    """Map repo names to their local clone paths."""
    base = Path("./.repo_cache/repos")
    mapping = {
        "huggingface/transformers": base / "huggingface__transformers",
        "pallets/flask": base / "flask",
        "psf/requests": base / "requests",
        "redis/redis": base / "redis",
        "sqlalchemy/sqlalchemy": base / "sqlalchemy",
    }
    result = {}
    for repo, path in mapping.items():
        if path.exists():
            result[repo] = str(path)
    return result


def get_existing_repos() -> set:
    """Read ChromaDB summary collection to find already-indexed repos."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        ef = ONNXMiniLM_L6_V2()
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        summary_col = client.get_collection("summary", embedding_function=ef)
        all_data = summary_col.get(limit=10000)
        repos = set()
        for m in all_data['metadatas']:
            if m and 'repo' in m:
                repos.add(m['repo'])
        return repos
    except Exception as e:
        print(f"  [警告] 读取已有仓库列表失败: {e}")
        return set()


def print_plan(existing: set):
    """Show what will be added."""
    by_lang: Dict[str, list] = {}
    est_total = 0
    for repo, _, lang in NEW_REPOS:
        by_lang.setdefault(lang, []).append(repo)
        if repo in existing:
            est_total += 0
        else:
            est_total += 1

    print("=" * 60)
    print("扩展计划: ChromaDB 新增仓库")
    print("=" * 60)
    print(f"已索引仓库数: {len(existing)}")
    print(f"新仓库总数: {est_total}")
    print()
    for lang, repos in by_lang.items():
        new = sum(1 for r in repos if r not in existing)
        exist = sum(1 for r in repos if r in existing)
        print(f"  [{lang}] {new} 新 / {exist} 已存在")
        for r in repos:
            skip = "(已存在)" if r in existing else ""
            print(f"    - {r} {skip}")
    print("=" * 60)


def clone_repo(repo: str, target: Path, timeout: int, max_retries: int = 2) -> bool:
    """Shallow clone."""
    if (target / ".git").exists():
        print(f"  [跳过] {repo} 已存在")
        return True
    for attempt in range(max_retries + 1):
        url = f"https://github.com/{repo}.git"
        print(f"  克隆 {repo} (timeout={timeout}s, attempt {attempt+1}/{max_retries+1})...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                print(f"  [OK] {repo} ({size/1024/1024:.0f}MB)")
                return True
            print(f"  [失败] {repo}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  [超时] {repo} ({timeout}s)")
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
        if attempt < max_retries:
            wait = 2 ** attempt * 5
            print(f"  等待 {wait}s 重试...")
            time.sleep(wait)
    return False


def extract_functions(source: str, ext: str) -> List[Dict]:
    """Extract function/class chunks by language."""
    if ext == ".py":
        chunks = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    text = ast.get_source_segment(source, node)
                    if text and len(text) > 30:
                        chunks.append({"type": "function", "name": node.name, "content": text})
                elif isinstance(node, ast.ClassDef):
                    text = ast.get_source_segment(source, node)
                    if text:
                        chunks.append({"type": "class", "name": node.name, "content": text})
        except SyntaxError:
            pass
        return chunks

    patterns = {
        ".java": [
            (r'(?:public|private|protected|static|final|abstract)\s+(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{[^}]*\}', 1),
            (r'(?:public|private|protected)\s+(?:static\s+)?(?:class|interface|enum)\s+(\w+)', 1),
        ],
        ".kt": [
            (r'(?:fun|suspend fun)\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
            (r'(?:class|interface|object|data class)\s+(\w+)', 1),
        ],
        ".kts": [
            (r'(?:fun|suspend fun)\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
            (r'(?:class|interface|object)\s+(\w+)', 1),
        ],
        ".ts": [
            (r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
            (r'(?:export\s+)?(?:default\s+)?(?:class|interface|type|enum)\s+(\w+)', 1),
            (r'(\w+)\s*[=:]\s*(?:async\s+)?\([^)]*\)\s*=>[^{]*\{[^}]*\}', 1),
        ],
        ".tsx": [
            (r'(?:export\s+)?(?:default\s+)?(?:class|interface|type|enum)\s+(\w+)', 1),
            (r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
        ],
        ".js": [
            (r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
            (r'(\w+)\s*=\s*(?:async\s+)?function\s*\([^)]*\)[^{]*\{[^}]*\}', 1),
            (r'(?:class|const)\s+(\w+)[^{]*\{[^}]*\}', 1),
        ],
    }

    chunks = []
    for pat, name_group in patterns.get(ext, []):
        for match in re.finditer(pat, source, re.DOTALL):
            text = match.group().strip()
            if len(text) < 30:
                continue
            name = match.group(name_group) if name_group <= (match.lastindex or 0) else f"m_{len(chunks)}"
            chunks.append({"type": "function", "name": name, "content": text})
    return chunks


def process_repo(repo: str, repo_path: str, lang: str) -> List[Dict]:
    """Chunk repo into code snippets."""
    root = Path(repo_path)
    chunks = []
    for root_dir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("node_modules", "__pycache__", "venv", "env", "target", "build", "dist")]
        for fname in files:
            fpath = Path(root_dir) / fname
            ext = fpath.suffix.lower()
            ext_lang = LANGUAGE_MAP.get(ext)
            if not ext_lang:
                continue
            if ext_lang == "javascript":  # skip pure JS for TS repos
                continue
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = fpath.relative_to(root).as_posix()
            funcs = extract_functions(source, ext)
            for fc in funcs:
                content_hash = hashlib.md5(fc["content"].encode()).hexdigest()[:12]
                chunks.append({
                    "repo": repo,
                    "language": ext_lang or lang,
                    "file_path": rel,
                    "name": fc["name"],
                    "type": fc["type"],
                    "content": fc["content"],
                    "hash": content_hash,
                })
    return chunks


def get_or_create_checked(client, ef, name):
    """Get collection; recreate if corrupted."""
    import gc
    for attempt in range(3):
        try:
            col = client.get_or_create_collection(name, embedding_function=ef)
            col.get(limit=1)  # verify readable
            return col
        except Exception as e:
            print(f"  [修复] {name} 集合损坏 (try {attempt+1}): {e}")
            try:
                client.delete_collection(name)
            except Exception:
                pass
            gc.collect()
    # Last try: create fresh
    return client.create_collection(name, embedding_function=ef)


def build_vector_db(all_chunks):
    """Add chunks to ChromaDB details collection. Creates file+repo summaries too."""
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    print("\n加载 ChromaDB...")
    ef = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    summary_col = get_or_create_checked(client, ef, "summary")
    details_col = get_or_create_checked(client, ef, "details")
    chunks_col = get_or_create_checked(client, ef, "chunks")

    # 去重: 跳过已有 content hash
    existing_hashes = set()
    try:
        all_meta = details_col.get(limit=200000)["metadatas"]
        for m in all_meta:
            if m and "hash" in m:
                existing_hashes.add(m["hash"])
    except Exception as e:
        print(f"  [跳过] 无法读取已有hash: {e}")

    dup_skip = 0
    new_items = []
    for c in all_chunks:
        if c["hash"] in existing_hashes:
            dup_skip += 1
            continue
        new_items.append(c)

    print(f"总解析: {len(all_chunks)}")
    print(f"去重跳过: {dup_skip}")
    print(f"新增: {len(new_items)}")

    if not new_items:
        print("无需新增.")
        return

    # 分批写入 details (函数/类级别)
    import gc
    batch_size = 64  # smaller batch to avoid ONNX memory issues
    batch_ids, batch_docs, batch_meta = [], [], []
    for i, c in enumerate(new_items):
        uid = f"v2_{i}_{c['repo']}|{c['file_path']}|{c['name']}|{hashlib.md5(c['content'].encode()).hexdigest()[:6]}"
        uid = uid.replace(":", "_").replace("|", "_")[:200]
        batch_ids.append(uid)
        batch_docs.append(c["content"])
        batch_meta.append({
            "repo": c["repo"], "language": c["language"],
            "file_path": c["file_path"], "name": c["name"],
            "type": c["type"], "level": 4, "hash": c["hash"],
        })
        if len(batch_ids) >= batch_size:
            details_col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            batch_ids, batch_docs, batch_meta = [], [], []
            gc.collect()
        if (i + 1) % 1000 == 0:
            print(f"  写入 {i+1}/{len(new_items)}...")

    if batch_ids:
        details_col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)

    # 也加入文件级摘要到 chunks
    file_map: Dict[str, List[Dict]] = {}
    for c in new_items:
        key = f"{c['repo']}|{c['file_path']}"
        file_map.setdefault(key, []).append(c)

    file_batch_ids, file_batch_docs, file_batch_meta = [], [], []
    for key, items in file_map.items():
        repo, fpath = key.split("|", 1)
        first = items[0]
        summary = f"# {fpath}\n" + "\n".join(
            f"  {c['type']}: {c['name']}" for c in items[:20]
        )
        uid = f"file_{repo}_{fpath.replace('/', '_')}_{hashlib.md5(key.encode()).hexdigest()[:8]}"
        uid = uid.replace(":", "_").replace("|", "_")[:150]
        file_batch_ids.append(uid)
        file_batch_docs.append(summary)
        file_batch_meta.append({
            "repo": repo, "language": first["language"],
            "file_path": fpath, "type": "file", "level": 3,
        })
        if len(file_batch_ids) >= batch_size:
            chunks_col.add(ids=file_batch_ids, documents=file_batch_docs, metadatas=file_batch_meta)
            file_batch_ids, file_batch_docs, file_batch_meta = [], [], []
            gc.collect()

    if file_batch_ids:
        chunks_col.add(ids=file_batch_ids, documents=file_batch_docs, metadatas=file_batch_meta)

    # 加入仓库级摘要到 summary
    repo_summaries = {}
    for c in new_items:
        repo_summaries.setdefault(c["repo"], {"languages": set(), "total": 0, "files": set()})
        repo_summaries[c["repo"]]["languages"].add(c["language"])
        repo_summaries[c["repo"]]["total"] += 1
        repo_summaries[c["repo"]]["files"].add(c["file_path"])

    summary_batch_ids, summary_batch_docs, summary_batch_meta = [], [], []
    for repo, info in repo_summaries.items():
        uid = f"repo_v2_{hashlib.md5(repo.encode()).hexdigest()[:12]}"
        summary = f"Repository: {repo}\nLanguages: {', '.join(sorted(info['languages']))}\nFiles: {len(info['files'])}\nChunks: {info['total']}"
        summary_batch_ids.append(uid)
        summary_batch_docs.append(summary)
        summary_batch_meta.append({"repo": repo, "name": repo, "type": "repo", "level": 1})
        if len(summary_batch_ids) >= batch_size:
            summary_col.add(ids=summary_batch_ids, documents=summary_batch_docs, metadatas=summary_batch_meta)
            summary_batch_ids, summary_batch_docs, summary_batch_meta = [], [], []
            gc.collect()

    if summary_batch_ids:
        summary_col.add(ids=summary_batch_ids, documents=summary_batch_docs, metadatas=summary_batch_meta)

    print(f"\n  details 新增:  {len(new_items)}")
    try:
        print(f"  details 总数:  {details_col.count()}")
    except Exception:
        pass


def load_chromadb_and_add(chunks):
    """Load ChromaDB and add chunks. Handles corrupted collections."""
    return build_vector_db(chunks)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    clone_only = "--clone-only" in sys.argv
    chunk_only = "--chunk-only" in sys.argv
    rebuild = "--rebuild" in sys.argv

    existing = get_existing_repos()
    print_plan(existing)

    if dry_run:
        sys.exit(0)

    # Step 1: Clone new repos (or use existing)
    cloned = {}
    if chunk_only:
        print("\n=== Step 1: 跳过克隆, 使用已下载仓库 ===")
        for repo, timeout, lang in NEW_REPOS:
            if repo in existing:
                continue
            safe = repo.replace("/", "__")
            target = NEW_REPOS_DIR / safe
            if (target / ".git").exists():
                cloned[repo] = {"path": str(target), "ok": True, "lang": lang}
                print(f"  [使用已有] {repo}")
            else:
                print(f"  [跳过] {repo} (未下载)")
    else:
        print("\n=== Step 1: 克隆新仓库 ===")
        for repo, timeout, lang in NEW_REPOS:
            if repo in existing:
                print(f"  [跳过] {repo} 已在向量库中")
                continue
            safe = repo.replace("/", "__")
            target = NEW_REPOS_DIR / safe
            ok = clone_repo(repo, target, timeout)
            cloned[repo] = {"path": str(target), "ok": ok, "lang": lang}

        if clone_only:
            ok = sum(1 for v in cloned.values() if v["ok"])
            print(f"\n克隆完成: {ok}/{len(cloned)} 成功")
            sys.exit(0)

    # Phase 3 repos for --rebuild (details collection recovery)
    if rebuild:
        print("\n=== [--rebuild] 同时重建 Phase 3 details ===")
        phase3_paths = get_local_phase3_paths()
        for repo, timeout, lang in PHASE3_REPOS:
            if repo not in cloned:
                path = phase3_paths.get(repo)
                if path:
                    cloned[repo] = {"path": path, "ok": True, "lang": lang}
                    print(f"  [添加 Phase 3] {repo}")

    # Step 2+3: parse + embed per-repo (isolate failures)
    import gc
    total_ok, total_fail = 0, 0
    for repo, info in cloned.items():
        if not info["ok"]:
            continue
        print(f"\n{'='*50}")
        print(f"  处理 {repo} ({info['lang']})...")
        print(f"{'='*50}")
        try:
            chunks = process_repo(repo, info["path"], info["lang"])
            print(f"  -> {len(chunks)} 代码块")
            if not chunks:
                print(f"  [跳过] 无代码块")
                continue
            load_chromadb_and_add(chunks)
            total_ok += 1
            print(f"  [完成] {repo} ({len(chunks)} 块)")
        except Exception as e:
            total_fail += 1
            print(f"  [错误] {repo}: {e}")
            import traceback
            traceback.print_exc()
        gc.collect()

    print(f"\n{'='*50}")
    print(f"全部完成: {total_ok} 成功, {total_fail} 失败")
    print(f"{'='*50}")

    print(f"\n完成! 向量库已扩展: {VECTOR_DB_DIR}/")
