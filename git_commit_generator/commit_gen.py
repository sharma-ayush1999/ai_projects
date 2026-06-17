import subprocess
import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
AI_MODEL=os.environ.get("AI_MODEL_DEV")

def get_git_diff():
    result = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output = True,
        text = True
    )
    return result.stdout

def generate_commit_message(diff):
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role":"system",
             "content": """You are an expert at writing git commit messages.
Given a git diff, generate a commit message following conventional commits format.

Rules:
- First line: <type>(<scope>): <short summary> (max 72 chars)
- Blank line
- Bullet points explaining what changed and why
- Types: feat, fix, docs, refactor, chore, style, test

Respond with JSON: {"title": "...", "body": "..."}"""
             }, 
             {
                 "role": "user",
                 "content": f"Generate a commit message for this diff:\n\n{diff}"
             }
        ], 
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

diff = get_git_diff()

if not diff:
    print("No staged changes found. Run git add <file>")
else:
    result = generate_commit_message(diff)
    print(f"\n{result['title']}\n")
    print(result['body'])