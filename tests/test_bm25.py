"""Tests for BM25 keyword search and Reciprocal Rank Fusion."""

from rag_app.models import RetrievedChunk
from rag_app.retrieval.bm25 import BM25Index, reciprocal_rank_fusion


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, metadata={"source_file": "test.txt"})


class TestBM25Index:
    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        idx.build([])
        assert idx.search("hello") == []

    def test_single_document_match(self):
        idx = BM25Index()
        idx.build([_chunk("c1", "The CAN FD protocol uses NBRP and DBRP.")])
        results = idx.search("NBRP")
        assert len(results) == 1
        assert results[0].id == "c1"

    def test_exact_identifier_ranked_first(self):
        idx = BM25Index()
        idx.build([
            _chunk("c1", "General automotive safety overview document."),
            _chunk("c2", "Error code ERR080082 indicates a CAN timing fault."),
            _chunk("c3", "The ERR080082 register is documented in section 5.3."),
        ])
        results = idx.search("ERR080082", top_k=3)
        # Both docs with ERR080082 should rank above the generic one.
        result_ids = [r.id for r in results]
        assert "c2" in result_ids[:2]
        assert "c3" in result_ids[:2]

    def test_no_match_returns_empty(self):
        idx = BM25Index()
        idx.build([_chunk("c1", "hello world")])
        results = idx.search("ZZZZZ")
        assert results == []

    def test_top_k_limits_results(self):
        idx = BM25Index()
        chunks = [_chunk(f"c{i}", f"word{i} common text") for i in range(20)]
        idx.build(chunks)
        results = idx.search("common", top_k=5)
        assert len(results) == 5


class TestReciprocalRankFusion:
    def test_merge_two_lists(self):
        list_a = [_chunk("c1", "a"), _chunk("c2", "b"), _chunk("c3", "c")]
        list_b = [_chunk("c3", "c"), _chunk("c1", "a"), _chunk("c4", "d")]
        merged = reciprocal_rank_fusion(list_a, list_b)
        ids = [c.id for c in merged]
        # c1 and c3 appear in both lists, so should rank highest.
        assert "c1" in ids[:2]
        assert "c3" in ids[:2]

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_single_list(self):
        chunks = [_chunk("c1", "a"), _chunk("c2", "b")]
        merged = reciprocal_rank_fusion(chunks)
        assert len(merged) == 2

    def test_weights_let_one_list_dominate(self):
        # The two lists rank the same chunks in opposite orders. Unweighted
        # (or equally weighted) RRF ties them; a heavier weight on one list
        # makes that list's order win.
        list_a = [_chunk("c1", "a"), _chunk("c2", "b")]
        list_b = [_chunk("c2", "b"), _chunk("c1", "a")]

        a_wins = reciprocal_rank_fusion(list_a, list_b, weights=[0.9, 0.1])
        assert [c.id for c in a_wins] == ["c1", "c2"]

        b_wins = reciprocal_rank_fusion(list_a, list_b, weights=[0.1, 0.9])
        assert [c.id for c in b_wins] == ["c2", "c1"]

    def test_equal_weights_match_unweighted_ranking(self):
        list_a = [_chunk("c1", "a"), _chunk("c2", "b"), _chunk("c3", "c")]
        list_b = [_chunk("c3", "c"), _chunk("c1", "a"), _chunk("c4", "d")]
        unweighted = [c.id for c in reciprocal_rank_fusion(list_a, list_b)]
        halved = [
            c.id
            for c in reciprocal_rank_fusion(list_a, list_b, weights=[0.5, 0.5])
        ]
        assert unweighted == halved

    def test_mismatched_weights_raise(self):
        import pytest

        with pytest.raises(ValueError):
            reciprocal_rank_fusion([_chunk("c1", "a")], weights=[0.5, 0.5])
