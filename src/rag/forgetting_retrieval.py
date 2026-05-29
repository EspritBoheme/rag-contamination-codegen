"""
遗忘感知检索模块
基于 FAISS 向量检索 + Ebbinghaus 时间衰减遗忘分数

公式:
  遗忘分数 = exp(-λ * Δt) * (1 + α * 访问次数) * β * 正确率
  检索重排: 最终分数 = 相似度 * (1 - 遗忘分数 * w)

用法:
  from forgetting_retrieval import ForgettingRetriever
  retriever = ForgettingRetriever(db_path="./vector_db")
  result = retriever.retrieve("binary search", top_k=3, use_forgetting=True)
"""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ============================================================
# 遗忘分数计算
# ============================================================
@dataclass
class AccessRecord:
    """检索条目的访问记录。"""
    last_access_time: float    # 最后访问时间 (unix timestamp)
    access_count: int = 0      # 访问次数
    correct_count: int = 0     # 正确使用次数
    total_use_count: int = 0   # 总使用次数
    created_time: float = 0.0  # 创建时间


class ForgettingScorer:
    """
    基于 Ebbinghaus 遗忘曲线的遗忘分数计算器。

    遗忘分数越高 = 越容易被遗忘 = 应该被过滤掉
    """

    def __init__(
        self,
        lambda_decay: float = 0.1,    # 时间衰减系数
        alpha_frequency: float = 0.3,  # 访问频率权重
        beta_correctness: float = 0.5, # 正确率权重
    ):
        self.lambda_decay = lambda_decay
        self.alpha_frequency = alpha_frequency
        self.beta_correctness = beta_correctness
        self.records: dict[str, AccessRecord] = {}

    def load_records(self, path: str):
        """加载访问记录。"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, val in data.items():
                self.records[key] = AccessRecord(**val)
            print(f"  加载 {len(self.records)} 条访问记录")

    def save_records(self, path: str):
        """保存访问记录。"""
        data = {}
        for key, rec in self.records.items():
            data[key] = {
                "last_access_time": rec.last_access_time,
                "access_count": rec.access_count,
                "correct_count": rec.correct_count,
                "total_use_count": rec.total_use_count,
                "created_time": rec.created_time,
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def score(self, doc_id: str, current_time: Optional[float] = None) -> float:
        """
        计算遗忘分数 (0~1)。
        0 = 记忆清晰, 1 = 完全遗忘
        """
        if doc_id not in self.records:
            return 0.5  # 未知条目, 中等遗忘分数

        rec = self.records[doc_id]
        if current_time is None:
            current_time = time.time()

        # 时间衰减: exp(-λ * Δt)
        dt = current_time - rec.last_access_time
        time_decay = math.exp(-self.lambda_decay * dt / 86400)  # 按天计算

        # 访问频率因子: 1 / (1 + α * 访问次数)
        frequency_factor = 1.0 / (1.0 + self.alpha_frequency * rec.access_count)

        # 正确率因子: 1 - 正确率
        if rec.total_use_count > 0:
            correctness = rec.correct_count / rec.total_use_count
        else:
            correctness = 0.5
        correctness_factor = 1.0 - correctness * self.beta_correctness

        # 综合遗忘分数
        forgetting_score = (1.0 - time_decay) * frequency_factor * correctness_factor
        return min(max(forgetting_score, 0.0), 1.0)

    def update_access(self, doc_id: str, correct: bool = True):
        """更新访问记录 (检索命中时调用)。"""
        if doc_id not in self.records:
            self.records[doc_id] = AccessRecord(
                last_access_time=time.time(),
                created_time=time.time(),
            )
        rec = self.records[doc_id]
        rec.last_access_time = time.time()
        rec.access_count += 1
        rec.total_use_count += 1
        if correct:
            rec.correct_count += 1


# ============================================================
# ChromaDB 检索器
# ============================================================
class ChromaRetriever:
    """基于 ChromaDB 的向量检索器。"""

    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = db_path
        self.client = None
        self.details = None
        self.chunks = None
        self._load_db()

    def _load_db(self):
        """加载 ChromaDB。"""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            ef = ONNXMiniLM_L6_V2()
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.details = self.client.get_collection("details", embedding_function=ef)
            self.chunks = self.client.get_collection("chunks", embedding_function=ef)
            print(f"  ChromaDB: details={self.details.count()}, chunks={self.chunks.count()}")
        except Exception as e:
            print(f"  [WARN] ChromaDB 加载失败: {e}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        向量检索。先搜 details, 不够再搜 chunks。

        Returns:
            [{"id": str, "text": str, "score": float, "metadata": dict}, ...]
        """
        if self.details is None:
            return []

        try:
            results = self.details.query(query_texts=[query], n_results=top_k)
            if not results["documents"][0]:
                results = self.chunks.query(query_texts=[query], n_results=top_k)

            docs = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append({
                    "id": meta.get("id", ""),
                    "text": doc[:500],
                    "score": float(1.0 - dist),  # ChromaDB distance -> similarity
                    "metadata": {
                        "repo": meta.get("repo", "?"),
                        "language": meta.get("language", "?"),
                    },
                })
            return docs
        except Exception as e:
            print(f"  [WARN] 检索失败: {e}")
            return []


# ============================================================
# 遗忘感知检索器 (整合)
# ============================================================
class ForgettingRetriever:
    """
    遗忘感知检索器。
    整合 FAISS 检索 + 遗忘分数计算 + 重排。
    """

    def __init__(
        self,
        db_path: str = "./vector_db",
        records_path: str = "./forgetting_records.json",
        forgetting_threshold: float = 0.7,
        forgetting_weight: float = 0.3,
    ):
        """
        Args:
            db_path: FAISS 索引路径
            records_path: 遗忘记录文件路径
            forgetting_threshold: 遗忘分数阈值 (高于此值被过滤)
            forgetting_weight: 遗忘分数在重排中的权重
        """
        self.forgetting_threshold = forgetting_threshold
        self.forgetting_weight = forgetting_weight

        # 初始化检索器
        self.retriever = ChromaRetriever(db_path)
        self.scorer = ForgettingScorer()
        self.scorer.load_records(records_path)
        self.records_path = records_path

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_forgetting: bool = True,
    ) -> dict:
        """
        遗忘感知检索。

        Args:
            query: 查询文本
            top_k: 返回条目数
            use_forgetting: 是否启用遗忘过滤

        Returns:
            {
                "context": str,          # 拼接后的检索上下文
                "total_retrieved": int,   # 原始检索条目数
                "filtered_count": int,    # 被遗忘机制过滤的数量
                "old_snippet_ratio": float,  # 旧代码片段占比
                "documents": list,        # 检索到的文档列表
            }
        """
        # 1. FAISS 检索 (多取一些用于过滤)
        fetch_k = top_k * 3 if use_forgetting else top_k
        raw_results = self.retriever.search(query, top_k=fetch_k)

        if not raw_results:
            return {
                "context": "",
                "total_retrieved": 0,
                "filtered_count": 0,
                "old_snippet_ratio": 0.0,
                "documents": [],
            }

        total_retrieved = len(raw_results)
        filtered_count = 0
        old_snippet_count = 0

        # 2. 计算遗忘分数并重排
        scored_results = []
        for doc in raw_results:
            doc_id = doc["id"]
            forgetting_score = self.scorer.score(doc_id)

            # 统计旧代码片段 (遗忘分数 > 0.5 视为"旧")
            if forgetting_score > 0.5:
                old_snippet_count += 1

            if use_forgetting:
                # 过滤: 遗忘分数高于阈值的条目被排除
                if forgetting_score >= self.forgetting_threshold:
                    filtered_count += 1
                    continue

                # 重排: 最终分数 = 相似度 * (1 - 遗忘分数 * 权重)
                final_score = doc["score"] * (1.0 - forgetting_score * self.forgetting_weight)
            else:
                final_score = doc["score"]

            scored_results.append({
                **doc,
                "forgetting_score": round(forgetting_score, 3),
                "final_score": round(final_score, 3),
            })

        # 3. 按最终分数排序, 取 top_k
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = scored_results[:top_k]

        # 4. 拼接上下文
        context_parts = []
        for doc in top_results:
            repo = doc["metadata"].get("repo", "?")
            lang = doc["metadata"].get("language", "?")
            sim = doc["final_score"]
            context_parts.append(f"--- [{lang}] {repo} (sim: {sim:.3f}) ---\n{doc['text'][:400]}")

        context = "\n\n".join(context_parts)

        # 5. 统计
        old_snippet_ratio = old_snippet_count / max(total_retrieved, 1)

        return {
            "context": context,
            "total_retrieved": total_retrieved,
            "filtered_count": filtered_count,
            "old_snippet_ratio": round(old_snippet_ratio, 3),
            "documents": top_results,
        }

    def save_records(self):
        """保存遗忘记录。"""
        self.scorer.save_records(self.records_path)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== 遗忘感知检索模块测试 ===\n")

    retriever = ForgettingRetriever(
        db_path="./vector_db",
        records_path="./forgetting_records.json",
        forgetting_threshold=0.7,
        forgetting_weight=0.3,
    )

    # 测试查询
    test_queries = [
        "binary search implementation",
        "LRU cache with O(1) operations",
        "merge k sorted linked lists",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        result = retriever.retrieve(query, top_k=3, use_forgetting=True)
        print(f"  检索到: {result['total_retrieved']} 条")
        print(f"  过滤: {result['filtered_count']} 条")
        print(f"  旧代码占比: {result['old_snippet_ratio']:.1%}")
        print(f"  上下文长度: {len(result['context'])} chars")

    retriever.save_records()
    print("\n完成!")
