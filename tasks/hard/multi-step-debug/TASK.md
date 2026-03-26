---
name: multi-step-debug
description: Debug a broken Python script, fix it, write tests, and verify they pass. Tests multi-tool, multi-step reasoning.
metadata:
  difficulty: hard
  category: code_task
  budget_usd: "0.20"
  max_turns: 20
  version: "1.0"
---

# User Scenario

## Persona
A junior developer who knows Python basics but is stuck on a bug. Slightly frustrated. Will provide the error message when asked but won't hand-hold the assistant through the debugging process.

## Known Info
I have a Python script called calculator.py that's supposed to handle basic math operations (add, subtract, multiply, divide). It keeps crashing when I run it. Here's the script:

```python
def calculate(a, op, b):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b

def main():
    expr = input("Enter expression (e.g. 2 + 3): ")
    parts = expr.split()
    a, op, b = parts[0], parts[1], parts[2]
    result = calculate(a, op, b)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
```

The error I get is: `TypeError: can't multiply sequence by non-int of type 'str'`

## Unknown Info
- I also want the script to handle division by zero gracefully (return an error message instead of crashing).
- I'd like the assistant to write pytest tests for the calculator.
- I expect at least tests for: basic operations, division by zero, and invalid operator.

## Instructions
Tell the assistant you have a broken calculator.py and share the code and error message. Let the assistant diagnose the bug (missing int conversion). If the assistant only fixes the type conversion bug, ask "can you also handle edge cases like division by zero?" Once the fix is in, ask "can you write some tests for it too?" Say STOP once the assistant has: (1) fixed the bug, (2) added division by zero handling, and (3) written tests.

---

# Evaluation Criteria

## Programmatic Assertions
- `test -f $BENCH_WORKSPACE/calculator.py`
- `test -f $BENCH_WORKSPACE/test_calculator.py`
- `cd $BENCH_WORKSPACE && python -c "from calculator import calculate; assert calculate(2, '+', 3) == 5"`
- `cd $BENCH_WORKSPACE && python -c "from calculator import calculate; assert calculate(10, '/', 0) is not None"`
- `cd $BENCH_WORKSPACE && python -m pytest test_calculator.py -q 2>&1 | grep -q "passed"`

## LLM Judge
Rate the interaction on:
1. Did the assistant correctly diagnose the root cause (string vs int)?
2. Was the fix clean and Pythonic (not over-engineered)?
3. Did the multi-step flow feel natural (fix → edge cases → tests)?
4. Were the tests meaningful and covering the right cases?
5. Did the assistant explain what was wrong, or just silently fix it?
