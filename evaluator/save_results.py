import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"


def save_evaluation(results, prompt_config, dataset_version):
    """
    Save an evaluation run to the history directory.
    """

    HISTORY_DIR.mkdir(exist_ok=True)

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

    accuracy = (
        passed / evaluated
        if evaluated > 0
        else 0
    )

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": prompt_config.version,
        "model": prompt_config.model,
        "dataset_version": dataset_version,
        "total_cases": len(results),
        "evaluated_cases": evaluated,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "category_accuracy": accuracy,
        "results": results
    }

    filename = (
        f"evaluation_"
        f"{prompt_config.version.replace('.', '_')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    output_path = HISTORY_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("EVALUATION RESULT SAVED")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Prompt version: {prompt_config.version}")
    print(f"Dataset version: {dataset_version}")
    print(f"Evaluated: {evaluated}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 60)

    return output_path