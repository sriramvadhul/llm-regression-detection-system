import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"


def load_evaluation(file_path):
    """Load an evaluation result JSON file."""

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def find_previous_evaluation(
    current_file
):
    """
    Find the most recent evaluation file
    before the current evaluation.
    """

    files = sorted(
        HISTORY_DIR.glob("evaluation_*.json")
    )

    previous_files = [
        file
        for file in files
        if file.resolve() != current_file.resolve()
    ]

    if not previous_files:
        return None

    return previous_files[-1]


def detect_regression(
    baseline,
    current,
    threshold=0.0
):
    """
    Compare baseline and current evaluation.

    A regression occurs when the current
    category accuracy decreases by more than
    the configured threshold.
    """

    baseline_accuracy = baseline[
        "category_accuracy"
    ]

    current_accuracy = current[
        "category_accuracy"
    ]

    accuracy_change = (
        current_accuracy
        - baseline_accuracy
    )

    regression = (
        accuracy_change < -threshold
    )

    return {
        "baseline_prompt_version":
            baseline["prompt_version"],

        "current_prompt_version":
            current["prompt_version"],

        "baseline_accuracy":
            baseline_accuracy,

        "current_accuracy":
            current_accuracy,

        "accuracy_change":
            accuracy_change,

        "regression_detected":
            regression
    }


def main():

    print("=" * 60)
    print("LLM REGRESSION DETECTION")
    print("=" * 60)

    evaluation_files = sorted(
        HISTORY_DIR.glob(
            "evaluation_*.json"
        )
    )

    if len(evaluation_files) < 2:

        print(
            "Not enough evaluation history."
        )

        print(
            "At least two evaluation runs "
            "are required for comparison."
        )

        return

    current_file = evaluation_files[-1]

    baseline_file = evaluation_files[-2]

    print(
        f"Baseline: {baseline_file.name}"
    )

    print(
        f"Current:  {current_file.name}"
    )

    print()

    baseline = load_evaluation(
        baseline_file
    )

    current = load_evaluation(
        current_file
    )

    result = detect_regression(
        baseline,
        current
    )

    print(
        f"Baseline prompt: "
        f"{result['baseline_prompt_version']}"
    )

    print(
        f"Current prompt:  "
        f"{result['current_prompt_version']}"
    )

    print()

    print(
        f"Baseline accuracy: "
        f"{result['baseline_accuracy']:.2%}"
    )

    print(
        f"Current accuracy:  "
        f"{result['current_accuracy']:.2%}"
    )

    print(
        f"Accuracy change:   "
        f"{result['accuracy_change']:+.2%}"
    )

    print()

    if result["regression_detected"]:

        print(
            "REGRESSION DETECTED"
        )

    else:

        print(
            "NO REGRESSION DETECTED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()