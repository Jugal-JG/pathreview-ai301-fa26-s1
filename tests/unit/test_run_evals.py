"""Tests for scripts/run_evals.py"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_evals.py"
spec = importlib.util.spec_from_file_location("run_evals", SCRIPT)
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)

PROFILE = {
    "github_username": "x",
    "repos": [
        {
            "name": "api",
            "description": "FastAPI service for transit data",
            "language": "Python",
            "topics": ["fastapi"],
            "readme_content": "# api\nA Python FastAPI service.",
        }
    ],
}


@pytest.mark.unit
class TestRunEvals:
    def test_writes_results_for_each_fixture(self, tmp_path):
        (tmp_path / "p.json").write_text(json.dumps(PROFILE))
        out = tmp_path / "eval_results.json"
        assert run_evals.main(tmp_path, out) == 0
        data = json.loads(out.read_text())
        assert len(data["profiles"]) == 1
        for key in ("relevance_score", "faithfulness_score", "overall_score"):
            assert 0.0 <= data["profiles"][0][key] <= 1.0
            assert key in data["mean"]

    def test_empty_fixtures_dir_fails_and_writes_nothing(self, tmp_path, capsys):
        out = tmp_path / "eval_results.json"
        assert run_evals.main(tmp_path, out) == 1
        assert not out.exists()
        assert "Evaluation failed" in capsys.readouterr().err

    def test_real_basic_profile_scores(self):
        profiles = dict(run_evals.load_profiles(run_evals.FIXTURES_DIR))
        row = run_evals.score_profile("basic_profile", profiles["basic_profile"])
        assert row["relevance_score"] > 0.0
