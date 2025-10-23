from openai import OpenAI
client = OpenAI(api_key='sk-AjgPUQzxcuKCscN3R0IPEru7G4hsAku16srLfzinmmn2AZKE', base_url="https://ai.sorasora.top/v1")

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "who are you?"},
    ],
    temperature=0.7,
    max_tokens=1000,
)

print(response)