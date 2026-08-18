import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "app"))


from classifier import classify_email
from prompt_loader import load_prompt


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

    return {
        "id": test_case["id"],
        "input": test_case["input"],
        "expected_category": expected_category,
        "actual_category": actual_category,
        "expected_summary": (
            test_case["expected_output"]["summary"]
        ),
        "actual_summary": result.summary,
        "category_pass": (
            actual_category == expected_category
        ),
        "difficulty": test_case["expected_difficulty"],
        "error": None
    }


def main():

    print("=" * 60)
    print("LLM REGRESSION EVALUATION")
    print("=" * 60)

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

            if result["category_pass"]:
                status = "PASS"
            else:
                status = "FAIL"

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
                f"    Status:   ERROR"
            )

            print(
                f"    Reason:   {error}"
            )

            results.append({
                "id": test_case["id"],
                "input": test_case["input"],
                "expected_category": (
                    test_case["expected_output"]["category"]
                ),
                "actual_category": None,
                "expected_summary": (
                    test_case["expected_output"]["summary"]
                ),
                "actual_summary": None,
                "category_pass": False,
                "difficulty": test_case["expected_difficulty"],
                "error": str(error)
            })

    # Calculate evaluation statistics

    passed = sum(
        1
        for result in results
        if result.get("category_pass") is True
        and result.get("error") is None
    )

    failed = sum(
        1
        for result in results
        if result.get("category_pass") is False
        and result.get("error") is None
    )

    errors = sum(
        1
        for result in results
        if result.get("error") is not None
    )

    evaluated = passed + failed

    if evaluated > 0:
        accuracy = passed / evaluated
    else:
        accuracy = 0

    # Final summary

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total test cases : {len(test_cases)}"
    )

    print(
        f"Evaluated        : {evaluated}"
    )

    print(
        f"Passed           : {passed}"
    )

    print(
        f"Failed           : {failed}"
    )

    print(
        f"Errors           : {errors}"
    )

    print(
        f"Category accuracy: {accuracy:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()