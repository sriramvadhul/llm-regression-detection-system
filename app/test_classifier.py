from prompt_loader import load_prompt
from classifier import classify_email


prompt = load_prompt(
    "prompts/support_classifier_v1.yaml"
)


email = """
Hi,

I was charged twice for my subscription this month.
Could you please check and refund the duplicate payment?

Thanks.
"""


result = classify_email(
    email=email,
    prompt_config=prompt
)


print("Category:", result.category)
print("Summary:", result.summary)