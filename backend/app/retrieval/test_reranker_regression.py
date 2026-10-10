import unittest
import numpy as np
from typing import List, Dict, Any
from backend.app.retrieval.production_retriever import production_retriever

class TestRerankerRegression(unittest.TestCase):
    """
    Regression tests for BGE Reranker v2 M3 indexing and score mapping:
    - Empty candidates
    - Single candidate
    - Fewer results than requested
    - Full 30-candidate pool
    - Duplicate review texts / IDs
    - Malformed / None / empty string texts
    - Safe fallback to RRF score on unexpected model errors
    """

    def setUp(self):
        self.retriever = production_retriever

    def test_empty_candidates(self):
        query = "What are the food complaints?"
        candidates: List[Dict[str, Any]] = []
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 0)

    def test_single_candidate(self):
        query = "slow service delay"
        candidates = [{
            "review_id": "r1",
            "business": "Test Bistro",
            "text": "The waiter took 45 minutes to bring our water.",
            "rrf_score": 0.03
        }]
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 1)
        self.assertIn("reranker_score", result[0])
        self.assertIsInstance(result[0]["reranker_score"], float)

    def test_multiple_candidates_sorting_direction(self):
        query = "terrible food tasted bad"
        candidates = [
            {
                "review_id": "r_low",
                "business": "Bistro A",
                "text": "Great place with lovely music and clean tables.",
                "rrf_score": 0.02
            },
            {
                "review_id": "r_high",
                "business": "Bistro B",
                "text": "The food tasted terrible, completely spoiled and undercooked.",
                "rrf_score": 0.01
            }
        ]
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 2)
        # Highly relevant complaint review should have higher reranker score than irrelevant music review
        self.assertGreater(result[0]["reranker_score"], result[1]["reranker_score"])
        self.assertEqual(result[0]["review_id"], "r_high")

    def test_malformed_and_empty_texts(self):
        query = "rude server"
        candidates = [
            {"review_id": "r_empty", "business": "A", "text": "", "rrf_score": 0.01},
            {"review_id": "r_none", "business": "B", "text": None, "rrf_score": 0.02},
            {"review_id": "r_valid", "business": "C", "text": "The server was extraordinarily rude.", "rrf_score": 0.03}
        ]
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn("reranker_score", item)
            self.assertIsInstance(item["reranker_score"], float)

    def test_thirty_candidates_pool(self):
        query = "long wait times"
        candidates = [
            {
                "review_id": f"rev_{i}",
                "business": f"Restaurant {i}",
                "text": f"Review {i}: Waited {i*5} minutes for a table." if i % 2 == 0 else f"Review {i}: Great meal loved it.",
                "rrf_score": round(1.0 / (60 + i), 6)
            }
            for i in range(30)
        ]
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 30)
        # Verify scores are sorted in descending order
        scores = [item["reranker_score"] for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_reproduce_original_failure_and_verify_fix(self):
        """
        Reproduces the exact original error:
        'index 4569596787400911650 is out of bounds for dimension 0 with size 30'
        Demonstrates that unpatched indexing raises IndexError, while _safe_rerank handles it safely.
        """
        import struct
        # 4569596787400911650 is the IEEE-754 64-bit int representation of float 0.003231080713140059
        float_val = 0.003231080713140059
        raw_int_index = struct.unpack(">q", struct.pack(">d", float_val))[0]
        self.assertEqual(raw_int_index, 4569596787400911650)

        # In unpatched indexing, using raw integer bits crashes on 30-candidate pool
        pool_size_30 = [f"item_{i}" for i in range(30)]
        with self.assertRaises(IndexError):
            _ = pool_size_30[raw_int_index]

        # In patched _safe_rerank, the float score is correctly recognized and mapped as a float score
        candidates = [
            {"review_id": f"rev_{i}", "business": "B", "text": f"Review {i}", "rrf_score": 0.01}
            for i in range(30)
        ]
        class MockReturningFloatScore:
            def predict(self, pairs):
                return np.full((30,), float_val)

        orig_reranker = self.retriever.reranker
        try:
            self.retriever.reranker = MockReturningFloatScore()
            safe_results = self.retriever._safe_rerank("wait time", candidates)
            self.assertEqual(len(safe_results), 30)
            self.assertAlmostEqual(safe_results[0]["reranker_score"], round(float_val, 6), places=5)
        finally:
            self.retriever.reranker = orig_reranker

    def test_fallback_when_reranker_fails(self):
        """Simulates reranker model throwing an exception to ensure safe RRF score preservation."""
        class MockFailingReranker:
            def predict(self, pairs):
                raise RuntimeError("index 4569596787400911650 is out of bounds for dimension 0 with size 30")

        original_reranker = self.retriever.reranker
        try:
            self.retriever.reranker = MockFailingReranker()
            query = "test query"
            candidates = [
                {"review_id": "r1", "business": "A", "text": "sample text 1", "rrf_score": 0.05},
                {"review_id": "r2", "business": "B", "text": "sample text 2", "rrf_score": 0.02}
            ]
            result = self.retriever._safe_rerank(query, candidates)
            self.assertEqual(len(result), 2)
            # Reranker score should fallback to rrf_score without crash
            self.assertEqual(result[0]["reranker_score"], 0.05)
            self.assertEqual(result[1]["reranker_score"], 0.02)
        finally:
            self.retriever.reranker = original_reranker

    def test_evidence_affirmation_negation_and_idiom(self):
        """Tests that negation and idioms ('never get sick of') are rejected, while true illnesses are accepted."""
        query = "Did anyone get sick or food poisoning?"
        praise_idiom = "I love this pizza and never get sick of eating here every Friday!"
        negated_claim = "The food was a bit cold, but nobody got sick and we were fine."
        true_illness = "We had dinner here and my friend got sick with violent food poisoning."

        self.assertFalse(self.retriever._check_evidence_affirmation(query, praise_idiom))
        self.assertFalse(self.retriever._check_evidence_affirmation(query, negated_claim))
        self.assertTrue(self.retriever._check_evidence_affirmation(query, true_illness))

    def test_duplicate_candidates_safe_rerank(self):
        """Tests that candidates with identical texts or scores are stably processed."""
        query = "bad customer service"
        candidates = [
            {"review_id": "r1", "business": "A", "text": "Extremely rude staff.", "rrf_score": 0.05},
            {"review_id": "r2", "business": "A", "text": "Extremely rude staff.", "rrf_score": 0.05},
        ]
        result = self.retriever._safe_rerank(query, candidates)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["reranker_score"], result[1]["reranker_score"])

if __name__ == "__main__":
    unittest.main()
