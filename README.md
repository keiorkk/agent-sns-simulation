# Social Media Simulation Framework (Graduate Research Project)

This is a simulation framework for modeling how information and misinformation spread on social media using LLM-based agents. The project was developed as part of my research in graduate school.

## Features

- Multi-agent simulation of SNS environments (posts, follows, comments)
- Agent personalities defined from real or synthetic personas
- Fact-checking mechanisms (third-party, community-based, hybrid)
- Data stored in SQLite for analysis

## My Role

I implemented the main simulation logic and agent behavior models, including:

- `simulation.py`: Simulation loop and event handling
- `agent_user.py`: Defines agent actions and response behavior
- `fact_checker.py`: LLM-based fact-checking logic
- `configs/*.json`: Experiment setting templates

## How to Run

1. Create a `keys.py` with your OpenAI API key:
```python
OPENAI_API_KEY = "your-api-key"
# The API key shown here has been disabled and is no longer valid.
