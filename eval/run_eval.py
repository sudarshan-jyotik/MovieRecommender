"""Evaluation runner for the CineMatch conversational movie agent."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import AgentRunResult, MovieAgent
from agent.llm import LLMAuthError, OpenAICompatibleLLMClient
from agent.tools import MovieToolset
from recommender import MovieRecommender

DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVAL_RESULTS_MD = ROOT_DIR / "EVAL_RESULTS.md"


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load JSONL evaluation cases from disk."""
    cases: list[dict[str, Any]] = []
    for raw_line in cases_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def percentile(values: list[float], pct: float) -> float:
    """Compute a linear-interpolated percentile in milliseconds."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(ordered[lo], 2)
    frac = idx - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)


def check_constraint_violation(
    case: dict[str, Any],
    result: AgentRunResult,
    recommender: MovieRecommender,
) -> tuple[bool, bool, str | None]:
    """Return (is_checkable, violated, reason) for a case's must_not rules."""
    must_not = case.get("must_not") or {}
    excluded_genres = {str(g).strip().lower() for g in (must_not.get("excluded_genres") or []) if g}
    forbidden_strings = [str(s) for s in (must_not.get("forbidden_strings") or []) if s]

    if not excluded_genres and not forbidden_strings:
        return False, False, None

    if excluded_genres:
        for pick in result.reply.picks:
            try:
                movie = recommender.get_movie(pick.movie_id)
            except KeyError:
                continue
            movie_genres = {str(g).strip().lower() for g in (movie.get("genres") or [])}
            overlap = excluded_genres & movie_genres
            if overlap:
                return True, True, f"Pick '{pick.title}' has excluded genre(s): {sorted(overlap)}"

    if forbidden_strings:
        haystack = " ".join(
            [
                result.reply.message,
                result.reply.follow_up_question or "",
                *(p.title + " " + p.reason for p in result.reply.picks),
            ]
        ).lower()
        for token in forbidden_strings:
            if token.lower() in haystack:
                return True, True, f"Reply contained forbidden string: {token!r}"

    return True, False, None


def evaluate_case(
    case: dict[str, Any],
    agent: MovieAgent,
    recommender: MovieRecommender,
    prompt_version: str,
    model: str | None,
) -> dict[str, Any]:
    """Run a single evaluation case and return per-case metrics."""
    res = agent.run(
        user_message=str(case["prompt"]),
        history=case.get("history"),
        preferences=case.get("preferences"),
        prompt_version=prompt_version,
        model=model,
    )

    called_tools = [entry.name for entry in res.trace]
    expected_tools = list(case.get("expected_tools") or [])
    tool_checkable = len(expected_tools) > 0
    tool_hit = any(name in expected_tools for name in called_tools) if tool_checkable else True

    checkable_constraint, violated_constraint, violation_reason = check_constraint_violation(
        case, res, recommender
    )

    picks_returned = len(res.reply.picks)
    total_picks_generated = picks_returned + res.hallucinated_picks_dropped

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "prompt": case["prompt"],
        "expected_tools": expected_tools,
        "called_tools": called_tools,
        "tool_checkable": tool_checkable,
        "tool_hit": tool_hit,
        "schema_valid_first_try": res.schema_valid_first_try,
        "schema_valid_after_retry": res.schema_valid_first_try or res.schema_valid_after_retry,
        "used_fallback": res.used_fallback,
        "picks_returned": picks_returned,
        "hallucinated_picks_dropped": res.hallucinated_picks_dropped,
        "total_picks_generated": total_picks_generated,
        "title_mismatches": res.title_mismatches,
        "constraint_checkable": checkable_constraint,
        "constraint_violated": violated_constraint,
        "violation_reason": violation_reason,
        "latency_ms": res.total_latency_ms,
        "trace": [t.model_dump() for t in res.trace],
    }


def summarize_results(
    case_results: list[dict[str, Any]],
    prompt_version: str,
    model: str,
) -> dict[str, Any]:
    """Aggregate case-level results into summary metrics."""
    n = len(case_results)
    tool_cases = [c for c in case_results if c["tool_checkable"]]
    constraint_cases = [c for c in case_results if c["constraint_checkable"]]

    total_generated_picks = sum(c["total_picks_generated"] for c in case_results)
    total_hallucinated = sum(c["hallucinated_picks_dropped"] for c in case_results)
    total_tool_calls = sum(len(c["called_tools"]) for c in case_results)
    latencies = [float(c["latency_ms"]) for c in case_results]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "model": model,
        "total_cases": n,
        "tool_selection_accuracy": (
            round(sum(1 for c in tool_cases if c["tool_hit"]) / len(tool_cases), 4)
            if tool_cases
            else 0.0
        ),
        "schema_valid_first_try_rate": (
            round(sum(1 for c in case_results if c["schema_valid_first_try"]) / n, 4) if n else 0.0
        ),
        "schema_valid_after_retry_rate": (
            round(sum(1 for c in case_results if c["schema_valid_after_retry"]) / n, 4) if n else 0.0
        ),
        "hallucination_rate": (
            round(total_hallucinated / total_generated_picks, 4)
            if total_generated_picks
            else 0.0
        ),
        "constraint_violation_rate": (
            round(sum(1 for c in constraint_cases if c["constraint_violated"]) / len(constraint_cases), 4)
            if constraint_cases
            else 0.0
        ),
        "avg_tool_calls_per_case": round(total_tool_calls / n, 2) if n else 0.0,
        "p50_latency_ms": percentile(latencies, 50.0),
        "p95_latency_ms": percentile(latencies, 95.0),
    }


def append_eval_markdown(summary: dict[str, Any], md_path: Path = EVAL_RESULTS_MD) -> None:
    """Write or append an evaluation run row to EVAL_RESULTS.md."""
    header = (
        "# Evaluation Results\n\n"
        "Run the evaluation harness against a live OpenAI-compatible LLM provider via:\n\n"
        "```bash\n"
        "python eval/run_eval.py --prompt-version v1\n"
        "python eval/run_eval.py --prompt-version v2\n"
        "```\n\n"
        "| Timestamp (UTC) | Prompt | Model | Cases | Tool Acc | Schema (1st) | Schema (Retry) | Hallucination | Constraint Viol. | Avg Tools | p50 (ms) | p95 (ms) |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    row = (
        f"| {summary['timestamp_utc'][:19]}Z | {summary['prompt_version']} | `{summary['model']}` | "
        f"{summary['total_cases']} | {summary['tool_selection_accuracy']:.1%} | "
        f"{summary['schema_valid_first_try_rate']:.1%} | {summary['schema_valid_after_retry_rate']:.1%} | "
        f"{summary['hallucination_rate']:.1%} | {summary['constraint_violation_rate']:.1%} | "
        f"{summary['avg_tool_calls_per_case']:.2f} | {summary['p50_latency_ms']:.1f} | "
        f"{summary['p95_latency_ms']:.1f} |\n"
    )

    if not md_path.exists():
        md_path.write_text(header + row, encoding="utf-8")
        return

    existing_lines = [
        line
        for line in md_path.read_text(encoding="utf-8").splitlines()
        if "not yet run" not in line.lower()
    ]
    md_path.write_text("\n".join(existing_lines).rstrip() + "\n" + row, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CineMatch agent evaluation suite.")
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL env var.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases.")
    args = parser.parse_args()

    llm_client = OpenAICompatibleLLMClient(model=args.model)
    if not llm_client.api_key:
        print("Error: LLM_API_KEY is not set. Set LLM_API_KEY in .env or environment before running evals.", file=sys.stderr)
        return 1

    recommender = MovieRecommender()
    agent = MovieAgent(llm_client=llm_client, toolset=MovieToolset(recommender))

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: max(1, args.limit)]

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running {case['id']} ({case.get('category', '')})...")
        results.append(
            evaluate_case(
                case=case,
                agent=agent,
                recommender=recommender,
                prompt_version=args.prompt_version,
                model=llm_client.model,
            )
        )

    summary = summarize_results(results, prompt_version=args.prompt_version, model=llm_client.model)
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = DEFAULT_RESULTS_DIR / f"{ts_slug}_{args.prompt_version}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    append_eval_markdown(summary)
    print(f"Saved evaluation results to {out_path} and updated {EVAL_RESULTS_MD}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LLMAuthError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        raise SystemExit(1)

