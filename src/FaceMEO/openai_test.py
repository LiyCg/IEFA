import os
from openai import OpenAI

# import pdb;pdb.set_trace()
client = OpenAI(
    api_key="API-KEY"
)

messages = [
    {"role" : "system", "content" : "you are a helpful assistant."},
    {"role" : "user", "content" : "Tell me a joke"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = messages,
    temperature=0.7
)

print(response.choices[0].message.content)