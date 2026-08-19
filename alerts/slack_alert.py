import os
import requests
from dotenv import load_dotenv


load_dotenv()


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_alert(
    status,
    baseline_accuracy,
    current_accuracy,
    regressions,
    improvements,
    baseline_version,
    current_version
):
    if not SLACK_WEBHOOK_URL:
        raise ValueError(
            "SLACK_WEBHOOK_URL was not found in .env"
        )

    accuracy_change = (
        current_accuracy
        - baseline_accuracy
    )

    message = {
        "text": (
            f"LLM Regression Evaluation\n"
            f"Status: {status}\n"
            f"Prompt: v{baseline_version} → v{current_version}\n"
            f"Baseline accuracy: {baseline_accuracy:.2%}\n"
            f"Current accuracy: {current_accuracy:.2%}\n"
            f"Accuracy change: {accuracy_change:+.2%}\n"
            f"Regressions: {regressions}\n"
            f"Improvements: {improvements}"
        )
    }

    response = requests.post(
        SLACK_WEBHOOK_URL,
        json=message,
        timeout=10
    )

    response.raise_for_status()

    print("Slack alert sent successfully.")