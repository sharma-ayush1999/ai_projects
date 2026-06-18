# Phase 1 — LLM Fundamentals

## How LLMs Work
- **Next-token predictors** — given text, predict next token, repeat until done
- Text splits into **tokens** (~¾ of a word). "Hello world" = 2 tokens
- **Context window** = max tokens per call (Claude 200k, GPT-4o 128k). Prompt + history + response all eat from it
- **No persistent state** — model forgets everything between calls. Memory is your job

## Temperature
| Value | Behavior | Use when |
|-------|----------|----------|
| `0` | Deterministic, identical every run | JSON, code, data extraction |
| `1.0` | Varied output each run | Creative, writing tasks |

**Rule: always use `0` for backend work.**

## Chat Format
```python
messages = [
    {"role": "system",    "content": "..."},  # behavior contract — your main control lever
    {"role": "user",      "content": "..."},  # human input
    {"role": "assistant", "content": "..."},  # model reply (include for memory)
]
```

## System Prompts
Same user message, completely different output based on system prompt.

```python
# Code only
{"role": "system", "content": "You are a senior backend engineer. Answer with code examples only. No prose."}

# Simple explanation
{"role": "system", "content": "You are a teacher explaining to a 10 year old. Use simple words and fun analogies."}

# JSON API
{"role": "system", "content": "You are an API. Always respond with valid JSON only. No explanation, no markdown."}
```

Use system prompts to define: **persona**, **output format**, **constraints**, **tone**.

## Structured JSON Output
```python
# ❌ Fragile — model ignores instructions and wraps in markdown fences
raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

# ✅ Production way — enforce at API level
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[...],
    temperature=0,
    response_format={"type": "json_object"}  # guarantees valid JSON
)
```
**Never trust the model to follow formatting instructions alone.**

## Multi-turn Conversations (Memory)
Model has zero memory. You pass full history manually every call.

```python
messages = [{"role": "system", "content": "You are a backend engineer assistant."}]

# Each turn:
messages.append({"role": "user", "content": user_input})
# ... call API ...
messages.append({"role": "assistant", "content": reply})
# Pass full messages list on every call — this IS the memory
```

## API Key Safety
```python
# ❌ Never hardcode
client = Groq(api_key="gsk_abc123...")

# ✅ Always use env vars
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```
```bash
GROQ_API_KEY="your_key" python script.py
```

## Minimal Working Example
```python
import os, json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are an API. Respond with valid JSON only."},
        {"role": "user",   "content": "Give me 3 HTTP status codes with meaning and when to use them."}
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

data = json.loads(response.choices[0].message.content)
print(data)
```

## Project Built
**Git Commit Message Generator** — reads staged `git diff`, generates conventional commit messages via LLM, outputs JSON with title + bullet points.
