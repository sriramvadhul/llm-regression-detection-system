import json
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "app")
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "evaluator")
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from classifier import classify_email
from prompt_loader import load_prompt
from save_results import save_evaluation


# ============================================================
# FILE PATHS
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "golden_dataset_v1.json"
)

PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "support_classifier_v1_1.yaml"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the golden evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# EVALUATE SINGLE TEST CASE
# ============================================================

def evaluate_case(
    test_case,
    prompt_config
):
    """
    Run one test case through the LLM classifier.
    """

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

        "expected_category":
            expected_category,

        "actual_category":
            actual_category,

        "expected_summary":
            test_case["expected_output"]["summary"],

        "actual_summary":
            result.summary,

        "category_pass":
            category_pass,

        "difficulty":
            test_case["expected_difficulty"],

        "error":
            None
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("LLM REGRESSION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load prompt configuration
    # --------------------------------------------------------

    print(
        "Loading prompt configuration..."
    )

    prompt_config = load_prompt(
        PROMPT_PATH
    )

    print(
        f"Prompt version: "
        f"{prompt_config.version}"
    )

    print(
        f"Model: "
        f"{prompt_config.model}"
    )

    print()

    # --------------------------------------------------------
    # Run test cases
    # --------------------------------------------------------

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

            # Determine status

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
                "    Status:   ERROR"
            )

            print(
                f"    Reason:   {error}"
            )

            # Store the error instead of
            # silently ignoring the test case

            results.append({

                "id":
                    test_case["id"],

                "input":
                    test_case["input"],

                "expected_category":
                    test_case[
                        "expected_output"
                    ]["category"],

                "actual_category":
                    None,

                "expected_summary":
                    test_case[
                        "expected_output"
                    ]["summary"],

                "actual_summary":
                    None,

                "category_pass":
                    False,

                "difficulty":
                    test_case[
                        "expected_difficulty"
                    ],

                "error":
                    str(error)
            })

    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    passed = sum(
        1
        for result in results
        if result.get(
            "category_pass"
        ) is True
        and result.get(
            "error"
        ) is None
    )

    failed = sum(
        1
        for result in results
        if result.get(
            "category_pass"
        ) is False
        and result.get(
            "error"
        ) is None
    )

    errors = sum(
        1
        for result in results
        if result.get(
            "error"
        ) is not None
    )

    evaluated = (
        passed
        + failed
    )

    if evaluated > 0:

        accuracy = (
            passed
            / evaluated
        )

    else:

        accuracy = 0


    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print()

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total test cases : "
        f"{len(test_cases)}"
    )

    print(
        f"Evaluated        : "
        f"{evaluated}"
    )

    print(
        f"Passed           : "
        f"{passed}"
    )

    print(
        f"Failed           : "
        f"{failed}"
    )

    print(
        f"Errors           : "
        f"{errors}"
    )

    print(
        f"Category accuracy: "
        f"{accuracy:.2%}"
    )

    print("=" * 60)


    # ========================================================
    # SAVE RESULT TO HISTORY
    # ========================================================

    save_evaluation(
        results,
        prompt_config,
        dataset["dataset_version"]
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()