---
name: create-hello-file
description: Create a text file with specific content. Tests basic file creation tool use.
metadata:
  difficulty: easy
  category: file_management
  budget_usd: "0.25"
  max_turns: 10
  version: "1.0"
---

# User Scenario

## Persona
Non-technical user. Concise and direct. Doesn't use technical jargon.

## Known Info
I need a file called hello.txt with the text "Hello World" in my current directory.

## Unknown Info
I also want a second line with today's date in YYYY-MM-DD format. I'll only mention this if the assistant asks whether I need anything else in the file.

## Instructions
Ask the assistant to create hello.txt with "Hello World" in it.
If the assistant asks whether you need anything else, mention the date line.
If the assistant just creates the file without asking, that's fine too.
Say STOP once the file is created.

---

# Evaluation Criteria

## Programmatic Assertions
- `test -f $BENCH_WORKSPACE/hello.txt`
- `grep -q "Hello World" $BENCH_WORKSPACE/hello.txt`

## LLM Judge
Rate the interaction: Was the assistant efficient? Did it confirm the file was created? Did it ask if the user needed anything else (proactive helpfulness)?
