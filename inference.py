"""
inference.py — OpenEnv AI Interview Screening Agent
Strict adherence to stdout format and OpenAI client usage.
"""

import os
import sys
import json
from openai import OpenAI
from my_env import AIInterviewEnv, Action

# ─── Configuration ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required to run inference.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASK_FILES = {
    "resume_screening_easy": "tasks/easy.json",
    "interview_evaluation_medium": "tasks/medium.json",
    "full_hiring_pipeline_hard": "tasks/hard.json",
}

# Allow setting BENCHMARK via env var, matching USER_REQUEST examples
BENCHMARK = os.environ.get("MY_ENV_BENCHMARK", "ai-interview-screening")

SYSTEM_PROMPT = """You are an AI hiring agent. Respond only with a JSON action.
Actions:
- {"action_type": "shortlist_candidate"}
- {"action_type": "reject_candidate"}
- {"action_type": "ask_question", "question": "..."}
- {"action_type": "evaluate_answer", "evaluation_score": 0.0-1.0}
- {"action_type": "hire_candidate"}
- {"action_type": "reject_after_interview"}

Follow instructions in the observation."""

def run_task(task_name: str, task_file: str):
    if not os.path.exists(task_file):
        return

    with open(task_file, "r") as f:
        task_config = json.load(f)

    env = AIInterviewEnv()
    obs = env.reset(task_config["config"])

    rewards = []
    steps = 0
    
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not env.done and steps < 20:
        obs_dict = obs.model_dump()
        obs_dict["current_stage"] = obs_dict["current_stage"].value
        
        error_msg = "null"
        action_obj = Action(action_type="__invalid__")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(obs_dict)}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_action = response.choices[0].message.content
            action_data = json.loads(raw_action)
            action_obj = Action(**action_data)
        except Exception as e:
            sanitized_error = str(e).replace('"', "").replace("\\", "")
            error_msg = f"'{sanitized_error}'"

        obs, reward, done = env.step(action_obj)
        steps += 1
        rewards.append(reward.value)

        # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        print(
            f"[STEP] step={steps} action={action_obj.action_type} "
            f"reward={reward.value:.2f} done={'true' if done else 'false'} "
            f"error={error_msg if error_msg != 'null' else 'null'}",
            flush=True
        )

    # Calculate final score and success
    success = rewards[-1] > 0 if rewards else False
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.0, min(1.0, score))
    
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True
    )

def main():
    # If a specific task is provided as an env var or arg, run only that
    target_task = os.environ.get("MY_ENV_TASK") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if target_task in TASK_FILES:
        run_task(target_task, TASK_FILES[target_task])
    else:
        for name, path in TASK_FILES.items():
            run_task(name, path)

if __name__ == "__main__":
    main()
