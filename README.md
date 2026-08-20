# LLM Regression Detection System

A CI/CD-style evaluation framework for detecting quality regressions in LLM-powered customer support classification when prompts or models change.

## Key Features

- Versioned prompts using YAML
- 50-case human-verified golden dataset
- Local LLM inference with Ollama + Llama 3.2 3B
- Deterministic structured JSON output with Pydantic
- Accuracy tracking across prompt versions
- Case-level regression and improvement detection
- Stable baseline comparisons
- Streamlit dashboard
- GitHub Actions quality gate
- Dockerized execution

## Results

| Prompt Version | Accuracy |
|---|---:|
| v1.0 | 82% |
| v1.1 | 86% |
| v1.2 | 90% |

Prompt v1.2 improved accuracy by 8 percentage points over v1.0 on the same 50-case golden dataset.

## Architecture

Customer Support Email
        ↓
Versioned Prompt
        ↓
Ollama / Llama 3.2 3B
        ↓
Pydantic Structured Output
        ↓
Golden Dataset Evaluation
        ↓
Historical Results
        ↓
Regression Detector
        ↓
Case-Level Comparison
        ↓
Streamlit Dashboard / CI Quality Gate

## Project Structure

app/
prompts/
datasets/
evaluator/
history/
baselines/
dashboard/
.github/workflows/
Dockerfile

## Run Locally

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt