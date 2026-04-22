from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api import create_app
from src.core.config import load_config


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        app = create_app(load_config())
        client = TestClient(app)

        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("model_exists", payload)
        self.assertIn("feature_columns_exists", payload)
        self.assertIn("metrics_exists", payload)


if __name__ == "__main__":
    unittest.main()
