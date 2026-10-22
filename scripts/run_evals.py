"""Run the RAG evaluation suite against benchmark portfolios.

This runner scores each fixture profile with the existing ``EvalSuite`` and
writes ``eval_results.json``. It does not run the generator: feedback is a fixed
reference review built from the fixture, so the scores check the wiring, not
review quality. Generated feedback needs a mock-LLM seam in ``ReviewGenerator``.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.evaluator.eval_suite import EvalSuite  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_profiles"
OUTPUT_PATH = REPO_ROOT / "eval_results.json"


def load_profiles(fixtures_dir: Path) -> list[tuple[str, dict]]:
    """Load every ``*.json`` profile in ``fixtures_dir``, sorted by filename.

    Args:
        fixtures_dir: Directory holding benchmark profile JSON files.

    Returns:
        List of (profile name, parsed profile dict) pairs.

    Raises:
        FileNotFoundError: If the directory holds no JSON files.
    """
    paths = sorted(Path(fixtures_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no benchmark profiles found in {fixtures_dir}")
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def build_inputs(profile: dict) -> tuple[str, list[dict], str]:
    """Build the (query, chunks, feedback) triple ``EvalSuite.run`` expects.

    Args:
        profile: Parsed benchmark profile with a ``repos`` list.

    Returns:
        Query text, one chunk per repo, and a fixed reference review.
    """
    repos = profile.get("repos", [])
    query_terms: list[str] = []
    chunks: list[dict] = []
    sentences: list[str] = []
    for repo in repos:
        query_terms.append(repo.get("language") or "")
        query_terms.extend(repo.get("topics", []))
        text = f"{repo.get('description', '')} {repo.get('readme_content', '')}".strip()
        chunks.append({"text": text})
        sentences.append(f"{repo.get('name', '')}: {repo.get('description', '')}")
    query = " ".join(t for t in query_terms if t)
    feedback = ". ".join(sentences)
    return query, chunks, feedback


def score_profile(name: str, profile: dict) -> dict:
    """Score one profile with ``EvalSuite``.

    Args:
        name: Profile name used in the report.
        profile: Parsed benchmark profile.

    Returns:
        Dict with the profile name and its three scores.
    """
    query, chunks, feedback = build_inputs(profile)
    result = EvalSuite().run(query, chunks, feedback)
    return {
        "profile": name,
        "relevance_score": result.relevance_score,
        "faithfulness_score": result.faithfulness_score,
        "overall_score": result.overall_score,
    }


def main(fixtures_dir: Path = FIXTURES_DIR, output_path: Path = OUTPUT_PATH) -> int:
    """Score every benchmark profile and write the JSON report.

    Args:
        fixtures_dir: Directory holding benchmark profile JSON files.
        output_path: Where to write the JSON report.

    Returns:
        Process exit code: 0 only if a report was written.
    """
    print("Running RAG evaluation suite...")
    try:
        profiles = load_profiles(fixtures_dir)
        rows = [score_profile(name, profile) for name, profile in profiles]
    except (OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    keys = ("relevance_score", "faithfulness_score", "overall_score")
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    Path(output_path).write_text(
        json.dumps({"profiles": rows, "mean": mean}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Scored {len(rows)} profile(s). Results written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
