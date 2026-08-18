import json
from pathlib import Path


DATASET_PATH = Path(__file__).parent / "golden_dataset_v1.json"

VALID_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "general"
}

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard"
}


def validate_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print(f"Dataset version: {dataset['dataset_version']}")

    test_cases = dataset["test_cases"]

    print(f"Number of test cases: {len(test_cases)}")

    ids = set()

    for case in test_cases:

        case_id = case["id"]

        # Check duplicate IDs
        if case_id in ids:
            raise ValueError(f"Duplicate test case ID: {case_id}")

        ids.add(case_id)

        # Check required fields
        required_fields = {
            "id",
            "input",
            "expected_output",
            "expected_difficulty",
            "notes"
        }

        missing_fields = required_fields - case.keys()

        if missing_fields:
            raise ValueError(
                f"{case_id} is missing fields: {missing_fields}"
            )

        # Validate category
        category = case["expected_output"]["category"]

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"{case_id} has invalid category: {category}"
            )

        # Validate difficulty
        difficulty = case["expected_difficulty"]

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"{case_id} has invalid difficulty: {difficulty}"
            )

        # Validate input
        if not case["input"].strip():
            raise ValueError(
                f"{case_id} has an empty input"
            )

        # Validate summary
        if not case["expected_output"]["summary"].strip():
            raise ValueError(
                f"{case_id} has an empty summary"
            )

    print("Dataset validation PASSED.")


if __name__ == "__main__":
    validate_dataset()