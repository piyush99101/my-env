from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import InterviewEnv
from env.models import Action

app = FastAPI()

# ---------------------------
# 📦 Request Models
# ---------------------------
class StepRequest(BaseModel):
    task: str
    answer: str


# ---------------------------
# 🌍 GLOBAL ENV STORE
# ---------------------------
env_store = {}


# ---------------------------
# 🔁 RESET
# ---------------------------
@app.post("/reset")
def reset(task: str):
    env = InterviewEnv(task)
    obs = env.reset()

    env_store[task] = env

    return {
        "question": obs.question
    }


# ---------------------------
# ▶️ STEP
# ---------------------------
@app.post("/step")
def step(req: StepRequest):
    env = env_store.get(req.task)

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    action = Action("respond", req.answer)
    result = env.step(action)

    return {
        "question": result.observation.question,
        "reward": float(result.reward),
        "done": result.done
    }


# ---------------------------
# 📊 STATE (OPTIONAL BUT SAFE)
# ---------------------------
@app.get("/state")
def state():
    return {"status": "running"}