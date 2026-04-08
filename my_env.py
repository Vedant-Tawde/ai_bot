import json
import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import uvicorn

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class Stage(str, Enum):
    SCREENING = "screening"
    INTERVIEW = "interview"
    EVALUATION = "evaluation"
    DECISION = "decision"
    DONE = "done"

class CandidateProfile(BaseModel):
    name: str
    skills: List[str]
    experience_years: int
    education: str
    projects: List[str]

class JobDescription(BaseModel):
    role: str
    required_skills: List[str]
    min_experience: int

class Observation(BaseModel):
    candidate_profile: Optional[CandidateProfile] = None
    job_description: JobDescription
    conversation_history: List[Dict[str, str]] = []
    current_stage: Stage
    step_count: int
    last_answer: Optional[str] = None
    instruction: str

class Action(BaseModel):
    action_type: str = Field(..., description="One of: shortlist_candidate, reject_candidate, ask_question, evaluate_answer, hire_candidate, reject_after_interview")
    question: Optional[str] = None
    evaluation_score: Optional[float] = None

class Reward(BaseModel):
    value: float
    description: str

# ─── Environment Logic ───────────────────────────────────────────────────────

class AIInterviewEnv:
    def __init__(self):
        self.reset()

    def reset(self, task_config: Dict[str, Any] = None):
        if task_config is None:
            # Default empty config for /reset ping
            task_config = {
                "candidate_profile": {
                    "name": "Default Candidate",
                    "skills": ["Python"],
                    "experience_years": 1,
                    "education": "BS",
                    "projects": []
                },
                "job_description": {
                    "role": "Intern",
                    "required_skills": ["Python"],
                    "min_experience": 0
                },
                "initial_stage": "screening"
            }
        
        self.candidate = CandidateProfile(**task_config["candidate_profile"])
        self.job = JobDescription(**task_config["job_description"])
        self.conversation_history = []
        self.current_stage = Stage(task_config.get("initial_stage", "screening"))
        self.step_count = 0
        self.last_answer = None
        self.done = False
        self.total_reward = 0.0
        
        return self.state()

    def state(self) -> Observation:
        return Observation(
            candidate_profile=self.candidate,
            job_description=self.job,
            conversation_history=self.conversation_history,
            current_stage=self.current_stage,
            step_count=self.step_count,
            last_answer=self.last_answer,
            instruction=self._get_instruction()
        )

    def _get_instruction(self) -> str:
        if self.current_stage == Stage.SCREENING:
            return "Screen the candidate resume against the job description. Action: 'shortlist_candidate' or 'reject_candidate'."
        elif self.current_stage == Stage.INTERVIEW:
            return "Conduct the interview. Ask a question relevant to the job requirements. Action: 'ask_question'."
        elif self.current_stage == Stage.EVALUATION:
            return f"Evaluate the candidate's last answer: '{self.last_answer}'. Action: 'evaluate_answer' with a score 0-1."
        elif self.current_stage == Stage.DECISION:
            return "Make a final hiring decision. Action: 'hire_candidate' or 'reject_after_interview'."
        return "Task completed."

    def step(self, action: Action) -> (Observation, Reward, bool):
        self.step_count += 1
        reward_val = 0.0
        reward_desc = "Neutral step"

        if self.current_stage == Stage.SCREENING:
            if action.action_type == "shortlist_candidate":
                is_good = self._is_candidate_qualified()
                reward_val = 0.2 if is_good else -0.2
                reward_desc = "Correctly shortlisted" if is_good else "Shortlisted unqualified candidate"
                self.current_stage = Stage.INTERVIEW
            elif action.action_type == "reject_candidate":
                is_good = self._is_candidate_qualified()
                if not is_good:
                    reward_val = 0.2
                    reward_desc = "Correctly rejected unqualified candidate"
                else:
                    reward_val = -0.5
                    reward_desc = "Wrongly rejected good candidate"
                self.current_stage = Stage.DONE
                self.done = True
            else:
                reward_val = -0.1
                reward_desc = f"Invalid action {action.action_type} for stage {self.current_stage}"

        elif self.current_stage == Stage.INTERVIEW:
            if action.action_type == "ask_question":
                is_relevant = any(skill.lower() in (action.question or "").lower() for skill in self.job.required_skills)
                reward_val = 0.3 if is_relevant else -0.2
                reward_desc = "Asked relevant question" if is_relevant else "Asked irrelevant question"
                self.last_answer = self._generate_answer(action.question or "")
                self.conversation_history.append({"role": "interviewer", "content": action.question or ""})
                self.conversation_history.append({"role": "candidate", "content": self.last_answer})
                self.current_stage = Stage.EVALUATION
            else:
                reward_val = -0.1
                reward_desc = "Invalid action"
        
        elif self.current_stage == Stage.EVALUATION:
            if action.action_type == "evaluate_answer":
                expected = self._get_expected_evaluation()
                if abs((action.evaluation_score or 0.0) - expected) < 0.2:
                    reward_val = 0.3
                    reward_desc = "Accurate evaluation"
                else:
                    reward_val = -0.2
                    reward_desc = "Inaccurate evaluation"
                self.current_stage = Stage.DECISION
            else:
                reward_val = -0.1
                reward_desc = "Invalid action"

        elif self.current_stage == Stage.DECISION:
            if action.action_type == "hire_candidate":
                is_qualified = self._is_candidate_qualified()
                reward_val = 1.0 if is_qualified else -0.5
                reward_desc = "Correct hire" if is_qualified else "Hired unqualified candidate"
                self.current_stage = Stage.DONE
                self.done = True
            elif action.action_type == "reject_after_interview":
                is_qualified = self._is_candidate_qualified()
                reward_val = 1.0 if not is_qualified else -0.5
                reward_desc = "Correct rejection" if not is_qualified else "Rejected qualified candidate"
                self.current_stage = Stage.DONE
                self.done = True
        
        self.total_reward += reward_val
        return self.state(), Reward(value=reward_val, description=reward_desc), self.done

    def _is_candidate_qualified(self) -> bool:
        has_skills = all(skill in self.candidate.skills for skill in self.job.required_skills)
        has_exp = self.candidate.experience_years >= self.job.min_experience
        return has_skills and has_exp

    def _generate_answer(self, question: str) -> str:
        question_lower = question.lower()
        matched = [sh for sh in self.candidate.skills if sh.lower() in question_lower]
        if matched:
            return f"I have worked with {matched[0]} extensively in my projects."
        return "I have basic knowledge but haven't used it much."

    def _get_expected_evaluation(self) -> float:
        if self.last_answer and any(s.lower() in self.last_answer.lower() for s in self.candidate.skills):
            return 1.0
        return 0.3

# ─── FastAPI Server ─────────────────────────────────────────────────────────

app = FastAPI()
env_instance = AIInterviewEnv()

@app.get("/")
async def root():
    return {
        "name": "ai-interview-screening-agent",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state"],
    }

@app.post("/reset")
async def reset(task_config: Dict[str, Any] = None):
    obs = env_instance.reset(task_config)
    return obs.model_dump()

@app.post("/step")
async def step(action: Action):
    obs, reward, done = env_instance.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done
    }

@app.get("/state")
async def state():
    return env_instance.state().model_dump()

def main():
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
