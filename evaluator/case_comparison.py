import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"

MODEL = "llama3.2:3b"

# Prompt versions to compare
BASELINE_VERSION = "1.1"
CURRENT_VERSION = "1.2"

MIN_COMPLETION_RATE = 0.90


def load_run(path):
    """Load an evaluation JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def completion_rate(run):
    """Calculate how much of the dataset was evaluated."""
    total = run.get("total_cases", 0)
    evaluated = run.get("evaluated_cases", 0)

    if total == 0:
        return 0

    return evaluated / total


def find_latest_run(prompt_version):
    """
    Find the latest valid evaluation run for
    the requested prompt version and model.
    """

    files = sorted(
        HISTORY_DIR.glob("evaluation_*.json"),
        reverse=True
    )

    for file in files:
        run = load_run(file)

        if (
            str(run.get("prompt_version")) == str(prompt_version)
            and run.get("model") == MODEL
            and completion_rate(run) >= MIN_COMPLETION_RATE
        ):
            return file, run

    return None, None


def build_result_map(run):
    """Create a dictionary indexed by test case ID."""
    return {
        item["id"]: item
        for item in run.get("results", [])
    }


def main():

    # ---------------------------------------------------------
    # Find baseline and current evaluation runs
    # ---------------------------------------------------------

    baseline_file, baseline = find_latest_run(
        BASELINE_VERSION
    )

    current_file, current = find_latest_run(
        CURRENT_VERSION
    )

    if baseline is None:
        print(
            f"No valid baseline found for "
            f"prompt {BASELINE_VERSION} "
            f"and model {MODEL}."
        )
        return

    if current is None:
        print(
            f"No valid current run found for "
            f"prompt {CURRENT_VERSION} "
            f"and model {MODEL}."
        )
        return

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    print("=" * 60)
    print("LLM CASE-LEVEL REGRESSION ANALYSIS")
    print("=" * 60)

    print(f"Model:    {MODEL}")
    print(f"Baseline: {baseline_file.name}")
    print(f"Current:  {current_file.name}")
    print()

    # ---------------------------------------------------------
    # Build result maps
    # ---------------------------------------------------------

    baseline_results = build_result_map(
        baseline
    )

    current_results = build_result_map(
        current
    )

    regressions = []
    improvements = []
    stable_pass = []
    stable_fail = []

    # ---------------------------------------------------------
    # Compare each test case
    # ---------------------------------------------------------

    for case_id, old in baseline_results.items():

        new = current_results.get(case_id)

        if new is None:
            continue

        old_pass = old.get(
            "category_pass",
            False
        )

        new_pass = new.get(
            "category_pass",
            False
        )

        # PASS -> FAIL
        if old_pass and not new_pass:

            regressions.append({
                "id": case_id,
                "input": new.get(
                    "input",
                    ""
                ),
                "expected": new.get(
                    "expected_category",
                    ""
                ),
                "baseline": old.get(
                    "actual_category",
                    ""
                ),
                "current": new.get(
                    "actual_category",
                    ""
                )
            })

        # FAIL -> PASS
        elif not old_pass and new_pass:

            improvements.append({
                "id": case_id,
                "input": new.get(
                    "input",
                    ""
                ),
                "expected": new.get(
                    "expected_category",
                    ""
                ),
                "baseline": old.get(
                    "actual_category",
                    ""
                ),
                "current": new.get(
                    "actual_category",
                    ""
                )
            })

        # PASS -> PASS
        elif old_pass and new_pass:

            stable_pass.append(
                case_id
            )

        # FAIL -> FAIL
        else:

            stable_fail.append(
                case_id
            )

    # ---------------------------------------------------------
    # Overall accuracy
    # ---------------------------------------------------------

    baseline_accuracy = baseline.get(
        "category_accuracy",
        0
    )

    current_accuracy = current.get(
        "category_accuracy",
        0
    )

    accuracy_change = (
        current_accuracy
        - baseline_accuracy
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("SUMMARY")
    print("-" * 60)

    print(
        f"Baseline accuracy : "
        f"{baseline_accuracy:.2%}"
    )

    print(
        f"Current accuracy  : "
        f"{current_accuracy:.2%}"
    )

    print(
        f"Accuracy change   : "
        f"{accuracy_change:+.2%}"
    )

    print()

    print(
        f"Regressions : "
        f"{len(regressions)}"
    )

    print(
        f"Improvements: "
        f"{len(improvements)}"
    )

    print(
        f"Stable pass : "
        f"{len(stable_pass)}"
    )

    print(
        f"Stable fail : "
        f"{len(stable_fail)}"
    )

    print()

    # ---------------------------------------------------------
    # Regressed cases
    # ---------------------------------------------------------

    if regressions:

        print("REGRESSED CASES")
        print("-" * 60)

        for item in regressions:

            print(
                f"ID:       "
                f"{item['id']}"
            )

            print(
                f"Expected: "
                f"{item['expected']}"
            )

            print(
                f"v{BASELINE_VERSION}:     "
                f"{item['baseline']}"
            )

            print(
                f"v{CURRENT_VERSION}:     "
                f"{item['current']}"
            )

            print(
                f"Input:    "
                f"{item['input']}"
            )

            print()

    else:

        print("REGRESSED CASES")
        print("-" * 60)
        print("No case-level regressions detected.")
        print()

    # ---------------------------------------------------------
    # Improved cases
    # ---------------------------------------------------------

    if improvements:

        print("IMPROVED CASES")
        print("-" * 60)

        for item in improvements:

            print(
                f"ID:       "
                f"{item['id']}"
            )

            print(
                f"Expected: "
                f"{item['expected']}"
            )

            print(
                f"v{BASELINE_VERSION}:     "
                f"{item['baseline']}"
            )

            print(
                f"v{CURRENT_VERSION}:     "
                f"{item['current']}"
            )

            print(
                f"Input:    "
                f"{item['input']}"
            )

            print()

    else:

        print("IMPROVED CASES")
        print("-" * 60)
        print("No case-level improvements detected.")
        print()

    # ---------------------------------------------------------
    # Stable failures
    # ---------------------------------------------------------

    if stable_fail:

        print("STABLE FAILED CASES")
        print("-" * 60)

        for case_id in stable_fail:

            old = baseline_results[case_id]
            new = current_results[case_id]

            print(f"ID:       {case_id}")

            print(
                f"Expected: "
                f"{new.get('expected_category', '')}"
            )

            print(
                f"v{BASELINE_VERSION}:     "
                f"{old.get('actual_category', '')}"
            )

            print(
                f"v{CURRENT_VERSION}:     "
                f"{new.get('actual_category', '')}"
            )

            print(
                f"Input:    "
                f"{new.get('input', '')}"
            )

            print()

    print("=" * 60)


if __name__ == "__main__":
    main()