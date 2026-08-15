from prompt_loader import load_prompt


prompt = load_prompt(
    "prompts/support_classifier_v1.yaml"
)

print("Prompt version:", prompt.version)
print("Model:", prompt.model)
print("System prompt:")
print(prompt.system_prompt)