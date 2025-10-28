import json
import re


def safe_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    raise ValueError("Failed to parse LLM JSON output.")


def llm_generate(prompt, client, model="openai/gpt-oss-120b", temp=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=temp,
    )
    return response.choices[0].message.content
