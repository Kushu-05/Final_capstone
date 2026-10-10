import os
import sys
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.app.main import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("active_llm_provider", data)
        self.assertIn("total_reviews", data)

    def test_02_config(self):
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("active_provider", data)
        self.assertTrue(data.get("local_fallback_available"))

    def test_03_businesses(self):
        resp = self.client.get("/api/businesses")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_04_search(self):
        payload = {
            "query": "slow service and cold food",
            "top_k": 3,
            "alpha": 0.5
        }
        resp = self.client.post("/api/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("results", data)
        self.assertIn("diagnostics", data)

    def test_05_csat_predict(self):
        payload = {"text": "The waiter was completely rude and the food was freezing cold. Horrible experience."}
        resp = self.client.post("/api/csat-predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predicted_stars", data)
        self.assertIn("satisfaction_tier", data)
        self.assertIn("nps_category", data)

    def test_06_trends(self):
        payload = {"business_id": "all", "category": "all"}
        resp = self.client.post("/api/trends", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("category_distribution", data)
        self.assertIn("historical_monthly_data", data)
        self.assertIn("overall_trajectory", data)

    def test_07_feedback_submission_and_retrieval(self):
        payload = {
            "query": "slow service",
            "category": "Service Speed",
            "is_helpful": True,
            "rating": 5,
            "revised_priority": "High",
            "manager_notes": "Implemented expedited ticket pass system.",
            "accepted_actions": ["Expedite orders over 15 mins"]
        }
        resp = self.client.post("/api/feedback", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        # Verify retrieved feedback
        get_resp = self.client.get("/api/feedback")
        self.assertEqual(get_resp.status_code, 200)
        feedback_data = get_resp.json()
        self.assertIn("metrics", feedback_data)
        self.assertIn("recent_feedback", feedback_data)

    def test_08_multi_agent_pipeline(self):
        payload = {
            "query": "long waiting time for food",
            "top_k": 3,
            "alpha": 0.5
        }
        resp = self.client.post("/api/multi-agent", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertIn("classification", data)
        self.assertIn("prioritization", data)
        self.assertIn("recommendations", data)
        self.assertIn("judge_audit", data)
        self.assertIn("workflow_steps", data)

if __name__ == "__main__":
    unittest.main()
