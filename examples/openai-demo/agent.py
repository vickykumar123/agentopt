"""A deliberately weak support agent — real Fireworks API (OpenAI-compatible).

Requires OPENAI_API_KEY in the environment.
"""

import os
import sys

from openai import OpenAI

SYSTEM_PROMPT = 'You are a support agent for FitCoach. When a customer asks about refunds, always mention that FitCoach offers a 30-day money-back guarantee.'

MODEL = "gpt-4o-mini"


def run(input_text):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        
    )
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text or ""},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(run(sys.argv[sys.argv.index("--input") + 1] if "--input" in sys.argv else "hi"))
