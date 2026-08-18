import json
import sys
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from app/
sys.path.insert(0, str(PROJECT_ROOT / "app"))


from classifier import classify_email
from prompt_loader import load_prompt


# Files
DATASET_PATH = PROJECT_ROOT / "datasets" / "golden_dataset_v1.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "support_classifier_v1_1.yaml"


def load_dataset():
    """Load the golden dataset."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(test_case, prompt_config):
    """Run one test case through Gemini."""

    result = classify_email(
        test_case["input"],
        prompt_config
    )

    expected_category = (
        test_case["expected_output"]["category"]
    )

    actual_category = result.category

    category_pass = (
        actual_category == expected_category
    )

    return {
        "id": test_case["id"],
        "input": test_case["input"],
        "expected_category": expected_category,
        "actual_category": actual_category,
        "expected_summary": (
            test_case["expected_output"]["summary"]
        ),
        "actual_summary": result.summary,
        "category_pass": category_pass,
        "difficulty": test_case["expected_difficulty"]
    }


def main():

    print("=" * 60)
    print("LLM REGRESSION EVALUATION")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset()

    test_cases = dataset["test_cases"]

    print(
        f"Dataset version: "
        f"{dataset['dataset_version']}"
    )

    print(
        f"Total test cases: "
        f"{len(test_cases)}"
    )

    print()

    # Load prompt configuration
    print("Loading prompt configuration...")

    prompt_config = load_prompt(PROMPT_PATH)

    print(
        f"Prompt version: "
        f"{prompt_config.version}"
    )

    print(
        f"Model: "
        f"{prompt_config.model}"
    )

    print()

    # Run evaluation
    results = []

    for index, test_case in enumerate(
        test_cases,
        start=1
    ):

        print(
            f"[{index}/{len(test_cases)}] "
            f"Running {test_case['id']}..."
        )

        try:

            result = evaluate_case(
                test_case,
                prompt_config
            )

            results.append(result)

            status = (
                "PASS"
                if result["category_pass"]
                else "FAIL"
            )

            print(
                f"    Expected: "
                f"{result['expected_category']}"
            )

            print(
                f"    Actual:   "
                f"{result['actual_category']}"
            )

            print(
                f"    Status:   "
                f"{status}"
            )

        except Exception as error:

            print(
                f"    ERROR: {error}"
            )

    # Calculate results
    passed = sum(
        1
        for result in results
        if result["category_pass"]
    )

    failed = len(results) - passed

    accuracy = (
        passed / len(results)
        if results
        else 0
    )

    # Summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total evaluated : {len(results)}"
    )

    print(
        f"Passed          : {passed}"
    )

    print(
        f"Failed          : {failed}"
    )

    print(
        f"Category accuracy: {accuracy:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()