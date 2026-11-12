"""Offline integration checks: run the real CLI and evaluator with a fake HTTP response."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temp.name)
        cls.repo = cls.work / "standalone checkout"
        cls.repo.mkdir()
        for name in ("run_benchmark.py", "verify_run.py"):
            shutil.copy2(ROOT / name, cls.repo / name)
        for name in ("benchmark_test", "data"):
            shutil.copytree(ROOT / name, cls.repo / name)
        cls.mock = cls.work / "mock"
        cls.mock.mkdir()
        (cls.mock / "sitecustomize.py").write_text('''
import json, os, threading
from pathlib import Path
import requests
request_lock = threading.Lock()
def post(url, *, json, headers, timeout):
    assert url == "https://offline.invalid/v1/chat/completions"
    assert headers["Authorization"] == "Bearer offline-test-key"
    assert json["model"] == "offline-model"
    assert json["seed"] == 42 and json["max_tokens"] == 1024
    assert "Target Gene" in json["messages"][0]["content"]
    with request_lock:
        with Path(os.environ["REQUEST_LOG"]).open("a", encoding="utf-8") as f:
            f.write("request\\n")
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content":
                '{"prediction":"up","reasoning":"Offline test response.","confidence":0.75}'}}]}
    return Response()
requests.post = post
''', encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_cli(self, run_id, mode="new", ratio="0.001", model="offline-model", workers="2"):
        log = self.work / (hashlib.sha256(run_id.encode()).hexdigest() + ".requests")
        log.write_text("", encoding="utf-8")
        env = dict(os.environ, OPENAI_API_KEY="offline-test-key", PYTHONUTF8="1",
                   PYTHONPATH=str(self.mock), REQUEST_LOG=str(log))
        result = subprocess.run([
            sys.executable, str(self.repo / "run_benchmark.py"), model,
            "https://offline.invalid/v1/chat/completions", run_id, workers, mode,
            "--sample-ratio", ratio,
        ], cwd=self.work, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.last_output = result.stdout + result.stderr
        return result, len(log.read_text(encoding="utf-8").splitlines())

    def result_path(self, run_id):
        return self.repo / "results" / f"{run_id}.jsonl"

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, self.last_output[-4000:])

    def test_01_new_run_and_no_overwrite(self):
        result, calls = self.run_cli("smoke")
        self.assert_success(result)
        self.assertEqual(calls, 10)
        path = self.result_path("smoke")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 10)
        stats = json.loads(path.with_name("smoke_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["total_samples"], 10)
        self.assertEqual(stats["max_workers"], 2)
        self.assertEqual(stats["correct"], sum(r["true_label"] == "up" for r in rows))
        self.assertFalse(Path(str(path) + ".partial").exists())
        self.assertFalse(Path(str(path) + ".runtime_seconds").exists())
        self.assertNotIn("offline-test-key", Path(str(path) + ".run.json").read_text())
        before = path.read_bytes()
        result, calls = self.run_cli("smoke")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_02_resume_and_torn_tail(self):
        result, _ = self.run_cli("seed")
        self.assert_success(result)
        seed = self.result_path("seed")
        lines = seed.read_text(encoding="utf-8").splitlines()
        for suffix, tail in (("newline", ""), ("torn", '\n{"sample_id":')):
            run_id = "resume_" + suffix
            path = self.result_path(run_id)
            shutil.copy2(str(seed) + ".run.json", str(path) + ".run.json")
            Path(str(path) + ".partial").write_text("\n".join(lines[:3]) + tail, encoding="utf-8")
            Path(str(path) + ".runtime_seconds").write_text("10.0", encoding="utf-8")
            result, calls = self.run_cli(run_id, "resume", workers="1")
            self.assert_success(result)
            self.assertEqual(calls, 7)
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(), lines)
            stats = json.loads(path.with_name(run_id + "_stats.json").read_text())
            self.assertEqual(stats["checkpoint_records_at_start"], 3)
            self.assertTrue(stats["elapsed_time_history_complete"])
            self.assertGreaterEqual(stats["elapsed_time"], 10)

    def test_03_reject_mismatched_resume(self):
        result, _ = self.run_cli("mismatch_seed")
        self.assert_success(result)
        seed = self.result_path("mismatch_seed")
        path = self.result_path("mismatch")
        shutil.copy2(str(seed) + ".run.json", str(path) + ".run.json")
        Path(str(path) + ".partial").write_text(seed.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        before = Path(str(path) + ".partial").read_bytes()
        for kwargs in ({"model": "different-model"}, {"ratio": "1.0"}):
            result, calls = self.run_cli("mismatch", "resume", **kwargs)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("configuration mismatch", self.last_output)
            self.assertEqual(calls, 0)
            self.assertEqual(Path(str(path) + ".partial").read_bytes(), before)

    def test_04_reject_corrupt_data_and_core(self):
        data = next((self.repo / "data/sampled_data").glob("*.jsonl"))
        core = self.repo / "benchmark_test/scripts/test_improved_cot.py"
        for path in (data, core):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                result, calls = self.run_cli("corrupt")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("checksum mismatch", self.last_output)
                self.assertEqual(calls, 0)
            finally:
                path.write_bytes(original)

    def test_05_full_10k(self):
        result, calls = self.run_cli("full", ratio="1.0", workers="8")
        self.assert_success(result)
        self.assertEqual(calls, 10000)
        stats = json.loads(self.result_path("full").with_name("full_stats.json").read_text())
        self.assertEqual(stats["total_samples"], 10000)
        self.assertEqual(len(stats["split_statistics"]), 8)
        self.assertEqual(stats["max_workers"], 8)

    def test_06_invalid_arguments(self):
        for kwargs in ({"workers": "0"}, {"ratio": "0"}, {"ratio": "nan"}):
            result, calls = self.run_cli("invalid", **kwargs)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls, 0)
        result, calls = self.run_cli("../escape")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
