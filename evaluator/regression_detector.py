import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"

MIN_COMPLETION_RATE = 0.90
WARNING_THRESHOLD = 0.03
CRITICAL_THRESHOLD = 0.08


def load_evaluation(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def completion_rate(run):
    total = run.get("total_cases", 0)
    evaluated = run.get("evaluated_cases", 0)

    if total == 0:
        return 0

    return evaluated / total


def validate_run(run, label):
    rate = completion_rate(run)

    if rate < MIN_COMPLETION_RATE:
        print(
            f"{label} run is incomplete: "
            f"{run.get('evaluated_cases', 0)}/"
            f"{run.get('total_cases', 0)} "
            f"({rate:.2%}) evaluated."
        )
        return False

    return True


def find_latest_run(prompt_version, model_name):
    files = sorted(
        HISTORY_DIR.glob("evaluation_*.json"),
        reverse=True
    )

    for file in files:
        run = load_evaluation(file)

        if (
            str(run.get("prompt_version")) == str(prompt_version)
            and run.get("model") == model_name
            and completion_rate(run) >= MIN_COMPLETION_RATE
        ):
            return file, run

    return None, None


def detect_regression(baseline, current):
    baseline_accuracy = baseline["category_accuracy"]
    current_accuracy = current["category_accuracy"]

    accuracy_change = current_accuracy - baseline_accuracy

    if accuracy_change <= -CRITICAL_THRESHOLD:
        status = "CRITICAL"

    elif accuracy_change <= -WARNING_THRESHOLD:
        status = "WARNING"

    else:
        status = "PASS"

    return {
        "baseline_accuracy": baseline_accuracy,
        "current_accuracy": current_accuracy,
        "accuracy_change": accuracy_change,
        "status": status
    }


def main():

    print("=" * 60)
    print("LLM REGRESSION DETECTION")
    print("=" * 60)

    MODEL = "llama3.2:3b"
    BASELINE_PROMPT = "1.1"
    CURRENT_PROMPT = "1.2"

    baseline_file, baseline = find_latest_run(
        BASELINE_PROMPT,
        MODEL
    )

    current_file, current = find_latest_run(
        CURRENT_PROMPT,
        MODEL
    )

    if baseline is None:
        print(
            f"No valid baseline run found for "
            f"prompt {BASELINE_PROMPT} and model {MODEL}."
        )
        return

    if current is None:
        print(
            f"No valid current run found for "
            f"prompt {CURRENT_PROMPT} and model {MODEL}."
        )
        return

    print(f"Baseline: {baseline_file.name}")
    print(f"Current:  {current_file.name}")
    print()

    print(
        f"Baseline model: "
        f"{baseline.get('model')}"
    )

    print(
        f"Current model:  "
        f"{current.get('model')}"
    )

    print(
        f"Baseline prompt: "
        f"{baseline.get('prompt_version')}"
    )

    print(
        f"Current prompt:  "
        f"{current.get('prompt_version')}"
    )

    print()

    if not validate_run(baseline, "Baseline"):
        return

    if not validate_run(current, "Current"):
        return

    result = detect_regression(
        baseline,
        current
    )

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

    print(
        f"Regression status: "
        f"{result['status']}"
    )
    if result["status"] == "CRITICAL":
        print("CI RESULT: FAIL")
        sys.exit(2)

    elif result["status"] == "WARNING":
        print("CI RESULT: FAIL")
        sys.exit(1)

    else:
        print("CI RESULT: PASS")
        sys.exit(0)
    print("=" * 60)


if __name__ == "__main__":
    main()