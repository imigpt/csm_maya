# app/services/gpt.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_response(prompt: str) -> str:
    try:
        print("🧠 Sending prompt to ChatGPT:", prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, friendly voice assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        print("💬 ChatGPT Reply:", reply)
        return reply
    except Exception as e:
        print("❌ ChatGPT Error:", e)
        return "Sorry, I had trouble generating a response."
