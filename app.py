from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import InterviewEnv
from env.models import Action

app = FastAPI()


# ---------------------------
# 🏠 ROOT (HEALTH CHECK)
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "AI Interview Trainer is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}

# ---------------------------
# 📦 Request Models
# ---------------------------
class StepRequest(BaseModel):
    task: str
    answer: str


class ResetRequest(BaseModel):
    task: str = "easy"


# ---------------------------
# 🌍 GLOBAL ENV STORE
# ---------------------------
env_store = {}


# ---------------------------
# 🔁 RESET
# ---------------------------
@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    env = InterviewEnv(req.task)
    obs = env.reset()

    env_store[req.task] = env

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


def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)