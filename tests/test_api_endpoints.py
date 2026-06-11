from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from src.api import create_app
from src.core.config import load_config

class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = load_config()
        self.app = create_app(cfg)
        self.client = TestClient(self.app)

    def test_health(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_evaluate_risk(self) -> None:
        payload = {
            "phone": "0912345678",
            "model_score": 0.8,
            "features": {
                "prefix": "091",
                "is_international": False,
                "total_call": 10,
                "miss_call": 2,
                "avg_duration": 45.0,
                "frequency_per_day": 2.5,
                "callback_rate": 0.5,
                "mostly_out_of_business_hour": False,
                "in_contact": False,
                "successful_call_count": 8,
                "avg_in_duration": 50.0
            }
        }
        resp = self.client.post("/evaluate/risk", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue("risk_evaluation" in data)
        self.assertTrue("final_level" in data["risk_evaluation"])

if __name__ == "__main__":
    unittest.main()
