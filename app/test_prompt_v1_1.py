import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from classifier import classify_email
from prompt_loader import load_prompt


DATASET_PATH = PROJECT_ROOT / "datasets" / "golden_dataset_v1.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "support_classifier_v1_1.yaml"


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)


prompt_config = load_prompt(PROMPT_PATH)

test_ids = ["TC008", "TC045"]

for test_case in dataset["test_cases"]:

    if test_case["id"] not in test_ids:
        continue

    print("=" * 60)
    print(f"Test case: {test_case['id']}")
    print(f"Input: {test_case['input']}")

    result = classify_email(
        test_case["input"],
        prompt_config
    )

    expected = test_case["expected_output"]["category"]
    actual = result.category

    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Summary:  {result.summary}")

    if expected == actual:
        print("Status:   PASS")
    else:
        print("Status:   FAIL")