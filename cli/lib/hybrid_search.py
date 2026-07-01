import os

from lib.InvertedIndex import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH, format_search_result, normalize, hybrid_score, rrf_score

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:

        results_bm25 = self._bm25_search(query, limit * 500)
        bm25_scores = [r[1] for r in results_bm25]  # r[1] is the score
        normalized_bm25 = normalize(bm25_scores)

        results_semantic = self.semantic_search.search_chunks(query, limit * 500)
        semantic_scores = [r["score"] for r in results_semantic]
        normalized_semantic = normalize(semantic_scores)
        semantic_score_map = {}

        for i, res in enumerate(results_semantic):
            semantic_score_map[res["id"]] = normalized_semantic[i]

        docID_dict = {}
        for i, res in enumerate(results_bm25):
            docID = res[0]
            docID_dict[docID] = (self.idx.docmap[docID], normalized_bm25[i], semantic_score_map.get(docID, 0.0), hybrid_score(normalized_bm25[i], semantic_score_map.get(docID, 0.0), alpha))
        results = []
        for doc_id, data in sorted(docID_dict.items(), key=lambda item: item[1][3], reverse=True):
            doc, bm25_norm, sem_norm, h_score = data
            results.append(format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["description"],
                score=h_score,
                bm25_score=bm25_norm,
                semantic_score=sem_norm,
            ))
        return results[:limit]


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        results_bm25 = self._bm25_search(query, limit * 500)
        results_semantic = self.semantic_search.search_chunks(query, limit * 500)

        semantic_ranks = {}
        for rank, item in enumerate(results_semantic, start=1):
            doc_id = item["id"]
            if doc_id not in semantic_ranks:
                semantic_ranks[doc_id] = rank

        bm25_ranks = {}
        for rank, item in enumerate(results_bm25, start=1):
            doc_id = item[0]
            if doc_id not in bm25_ranks:
                bm25_ranks[doc_id] = rank

        docID_dict = {}
        all_ids = set(bm25_ranks) | set(semantic_ranks)

        for docID in all_ids:
            bm25_rank = bm25_ranks.get(docID)      # None if absent
            sem_rank = semantic_ranks.get(docID)   # None if absent

            bm25_part = rrf_score(bm25_rank, k) if bm25_rank is not None else 0
            sem_part = rrf_score(sem_rank, k) if sem_rank is not None else 0
            total = bm25_part + sem_part

            docID_dict[docID] = (self.idx.docmap[docID], bm25_rank, sem_rank, total)
            
        results = []
        for doc_id, data in sorted(docID_dict.items(), key=lambda item: item[1][3], reverse=True):
            doc, bm25_norm, sem_norm, rf_score = data
            results.append(format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                score=rf_score,
                bm25_rank=bm25_norm,
                semantic_rank=sem_norm,
            ))
        return results[:limit]
