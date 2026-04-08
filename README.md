---
title: ai_bot
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI Interview Screening Agent — OpenEnv Submission

A production-ready OpenEnv environment simulating a multi-stage hiring pipeline.

## Motivation
Hiring is a high-stakes, multi-step process requiring analytical screening, strategic questioning, and objective evaluation. This environment models that complexity, providing a real-world task for evaluating agentic intelligence.

## Environment Design

### State & Observations
The environment provides a JSON observation at each step:
- **`candidate_profile`**: Resume data (skills, experience, projects).
- **`job_description`**: Requirement data.
- **`conversation_history`**: Record of all interview turns.
- **`current_stage`**: `screening` -> `interview` -> `evaluation` -> `decision`.

### Action Space
- `shortlist_candidate` / `reject_candidate`
- `ask_question(question: str)`
- `evaluate_answer(score: float)`
- `hire_candidate` / `reject_after_interview`

### Reward Shaping
- **Correct Progress**: +0.2 to +0.3 per correct intermediate step.
- **Success**: +1.0 for the correct final hiring/rejection decision.
- **Penalties**: -0.5 for hiring an unqualified candidate or rejecting a qualified one.

## Tasks
1. **Easy (Resume Screening)**: Single-step decision on a highly qualified candidate.
2. **Medium (Interview Evaluation)**: Screen and conduct a one-turn interview.
3. **Hard (Full Pipeline)**: Complete end-to-end process for a senior leadership role.

## Setup & Usage

### Local Development
```bash
pip install -r requirements.txt
cp .env.example .env
# Start the environment server
python my_env.py
# Run inference in another terminal
python inference.py
```

### Docker
```bash
docker build -t ai-interview-agent .
docker run -p 8080:8080 --env-file .env ai-interview-agent
```

## Baseline Scores (gpt-4o-mini)
- **Easy**: 1.00 success, ~0.20 score
- **Medium**: 1.00 success, ~0.45 score
- **Hard**: 1.00 success, ~0.45 score

## OpenEnv Tags
- `openenv`
- `real-world`
- `hiring-agent`
