import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"

MODEL = "llama3.2:3b"
BASELINE_VERSION = "1.1"
CURRENT_VERSION = "1.2"
MIN_COMPLETION_RATE = 0.90


st.set_page_config(
    page_title="LLM Regression Dashboard",
    layout="wide"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def completion_rate(run):
    total = run.get("total_cases", 0)
    evaluated = run.get("evaluated_cases", 0)

    if total == 0:
        return 0

    return evaluated / total


def load_runs():
    runs = []

    for file in sorted(
        HISTORY_DIR.glob("evaluation_*.json")
    ):
        data = load_json(file)

        runs.append({
            "file": file.name,
            "prompt_version": str(
                data.get("prompt_version")
            ),
            "model": data.get("model"),
            "accuracy": data.get(
                "category_accuracy",
                0
            ),
            "passed": data.get("passed", 0),
            "failed": data.get("failed", 0),
            "errors": data.get("errors", 0),
            "evaluated": data.get(
                "evaluated_cases",
                0
            ),
            "total": data.get(
                "total_cases",
                0
            ),
            "results": data.get(
                "results",
                []
            )
        })

    return runs


def find_latest_valid_run(
    runs,
    prompt_version
):
    valid_runs = []

    for run in runs:

        if (
            run["prompt_version"] == prompt_version
            and run["model"] == MODEL
        ):

            if run["total"] == 0:
                continue

            rate = (
                run["evaluated"]
                / run["total"]
            )

            if rate >= MIN_COMPLETION_RATE:
                valid_runs.append(run)

    if not valid_runs:
        return None

    return valid_runs[-1]


def build_result_map(run):
    return {
        result["id"]: result
        for result in run["results"]
    }


def compare_runs(
    baseline,
    current
):

    baseline_map = build_result_map(
        baseline
    )

    current_map = build_result_map(
        current
    )

    regressions = []
    improvements = []
    stable_pass = []
    stable_fail = []

    for case_id, old in baseline_map.items():

        new = current_map.get(case_id)

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

        if old_pass and not new_pass:

            regressions.append({
                "ID": case_id,
                "Input": new.get(
                    "input",
                    ""
                ),
                "Expected": new.get(
                    "expected_category",
                    ""
                ),
                f"v{BASELINE_VERSION}":
                    old.get(
                        "actual_category",
                        ""
                    ),
                f"v{CURRENT_VERSION}":
                    new.get(
                        "actual_category",
                        ""
                    )
            })

        elif not old_pass and new_pass:

            improvements.append({
                "ID": case_id,
                "Input": new.get(
                    "input",
                    ""
                ),
                "Expected": new.get(
                    "expected_category",
                    ""
                ),
                f"v{BASELINE_VERSION}":
                    old.get(
                        "actual_category",
                        ""
                    ),
                f"v{CURRENT_VERSION}":
                    new.get(
                        "actual_category",
                        ""
                    )
            })

        elif old_pass and new_pass:

            stable_pass.append(case_id)

        else:

            stable_fail.append(case_id)

    return {
        "regressions": regressions,
        "improvements": improvements,
        "stable_pass": stable_pass,
        "stable_fail": stable_fail
    }


# ============================================================
# LOAD DATA
# ============================================================

runs = load_runs()

st.title(
    "LLM Regression Detection Dashboard"
)

st.caption(
    "Track prompt quality, regressions, improvements, "
    "and failed golden-dataset cases."
)


if not runs:
    st.warning(
        "No evaluation history found."
    )
    st.stop()


baseline = find_latest_valid_run(
    runs,
    BASELINE_VERSION
)

current = find_latest_valid_run(
    runs,
    CURRENT_VERSION
)


# ============================================================
# LATEST COMPARISON
# ============================================================

if baseline and current:

    comparison = compare_runs(
        baseline,
        current
    )

    accuracy_change = (
        current["accuracy"]
        - baseline["accuracy"]
    )

    st.subheader(
        f"Prompt Comparison: "
        f"v{BASELINE_VERSION} → "
        f"v{CURRENT_VERSION}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        f"v{BASELINE_VERSION} Accuracy",
        f"{baseline['accuracy']:.2%}"
    )

    col2.metric(
        f"v{CURRENT_VERSION} Accuracy",
        f"{current['accuracy']:.2%}",
        delta=f"{accuracy_change:+.2%}"
    )

    col3.metric(
        "Regressions",
        len(
            comparison["regressions"]
        )
    )

    col4.metric(
        "Improvements",
        len(
            comparison["improvements"]
        )
    )

    st.divider()


# ============================================================
# ACCURACY TREND
# ============================================================

st.subheader(
    "Prompt Accuracy Trend"
)

trend_runs = [
    run
    for run in runs
    if (
        run["model"] == MODEL
        and run["errors"] == 0
        and run["total"] > 0
        and completion_rate({
            "total_cases": run["total"],
            "evaluated_cases": run["evaluated"]
        }) >= MIN_COMPLETION_RATE
    )
]


latest_by_version = {}

for run in trend_runs:
    latest_by_version[
        run["prompt_version"]
    ] = run


trend_data = []

for version, run in latest_by_version.items():

    trend_data.append({
        "Prompt Version":
            f"v{version}",
        "Accuracy":
            run["accuracy"] * 100
    })


trend_df = pd.DataFrame(
    trend_data
)

if not trend_df.empty:

    st.line_chart(
        trend_df.set_index(
            "Prompt Version"
        )
    )


# ============================================================
# REGRESSIONS AND IMPROVEMENTS
# ============================================================

if baseline and current:

    st.subheader(
        "Case-Level Changes"
    )

    regression_tab, improvement_tab = st.tabs(
        [
            "Regressions",
            "Improvements"
        ]
    )

    with regression_tab:

        regressions = (
            comparison["regressions"]
        )

        if regressions:

            st.dataframe(
                pd.DataFrame(
                    regressions
                ),
                use_container_width=True
            )

        else:

            st.success(
                "No case-level regressions."
            )

    with improvement_tab:

        improvements = (
            comparison["improvements"]
        )

        if improvements:

            st.dataframe(
                pd.DataFrame(
                    improvements
                ),
                use_container_width=True
            )

        else:

            st.info(
                "No case-level improvements."
            )


# ============================================================
# LATEST FAILED CASES
# ============================================================

if current:

    st.subheader(
        f"Failed Cases in Prompt "
        f"v{CURRENT_VERSION}"
    )

    failed_cases = [
        result
        for result in current["results"]
        if (
            result.get(
                "category_pass"
            ) is False
            and result.get(
                "error"
            ) is None
        )
    ]

    if failed_cases:

        failed_df = pd.DataFrame([
            {
                "ID":
                    case.get("id"),

                "Input":
                    case.get("input"),

                "Expected":
                    case.get(
                        "expected_category"
                    ),

                "Actual":
                    case.get(
                        "actual_category"
                    ),

                "Difficulty":
                    case.get(
                        "difficulty"
                    )
            }
            for case in failed_cases
        ])

        st.dataframe(
            failed_df,
            use_container_width=True
        )

    else:

        st.success(
            "No failed cases."
        )


# ============================================================
# EVALUATION HISTORY
# ============================================================

st.subheader(
    "Evaluation History"
)


history_df = pd.DataFrame([
    {
        "Prompt Version":
            f"v{run['prompt_version']}",

        "Model":
            run["model"],

        "Accuracy":
            f"{run['accuracy']:.2%}",

        "Passed":
            run["passed"],

        "Failed":
            run["failed"],

        "Errors":
            run["errors"],

        "Evaluated":
            f"{run['evaluated']}/{run['total']}",

        "File":
            run["file"]
    }
    for run in runs
])


st.dataframe(
    history_df,
    use_container_width=True
)