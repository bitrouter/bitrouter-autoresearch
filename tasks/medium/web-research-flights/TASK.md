---
name: web-research-flights
description: Research flight options from SFO to Tokyo, requiring multi-turn clarification of preferences.
metadata:
  difficulty: medium
  category: information_retrieval
  budget_usd: "0.50"
  max_turns: 15
  version: "1.0"
---

# User Scenario

## Persona
A busy professional planning a vacation. Responds quickly but with incomplete information. Expects the assistant to ask clarifying questions rather than making assumptions.

## Known Info
I want to fly from San Francisco (SFO) to Tokyo sometime in June 2026. My budget is under $1200 round trip.

## Unknown Info
- I'm flexible on dates within June 1-15.
- I strongly prefer direct flights but would accept one stop if significantly cheaper.
- I'm flying alone (one passenger).
- I have no airline loyalty preferences.

## Instructions
Ask the assistant to help you find flights to Tokyo. Give only the basic info (SFO, Tokyo, June, $1200 budget) initially. Reveal the additional details only when the assistant asks relevant questions. If the assistant presents a summary of options with prices, say STOP. If the assistant asks a question you don't have an answer to based on your scenario, say you're flexible or defer to their recommendation.

---

# Evaluation Criteria

## Programmatic Assertions
- `test -f $BENCH_WORKSPACE/flights.md || test -f $BENCH_WORKSPACE/flights.txt`

## LLM Judge
Rate the interaction on:
1. Did the assistant ask clarifying questions (dates, passengers, stops) before searching?
2. Were the flight options presented in a clear, comparable format?
3. Did the assistant respect the $1200 budget constraint?
4. Was the multi-turn conversation natural and efficient, or did it feel robotic?
