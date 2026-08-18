import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Say hello and tell me that you are connected successfully."
        }
    ]
)

print(response["message"]["content"])