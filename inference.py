import os
from openai import OpenAI
from env.environment import InterviewEnv
from env.models import Action

# ---------------------------
# 🔐 ENV CONFIG
# ---------------------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

# ---------------------------
# 🧠 FALLBACK AGENT (NO API)
# ---------------------------
def fallback_agent(question):
    q = question.lower()

    if "yourself" in q:
        return (
            "I am a student interested in technology because I enjoy building things. "
            "For example, I created small coding projects to improve my skills."
        )

    elif "strength" in q:
        return (
            "My strength is consistency because I practice regularly. "
            "For example, I improve my coding by building projects daily."
        )

    elif "hire" in q:
        return (
            "You should hire me because I am dedicated and quick to learn. "
            "For example, I have built projects that solve real problems."
        )

    elif "challenge" in q:
        return (
            "I faced a challenge while learning coding because it was difficult at first. "
            "For example, I struggled with debugging but improved by practicing daily."
        )

    return (
        "I am always learning because I want to improve myself. "
        "For example, I work on projects to gain experience."
    )


# ---------------------------
# 🤖 LLM AGENT
# ---------------------------
def llm_agent(question):
    if client is None:
        return fallback_agent(question)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a 15-year-old student preparing for interviews. "
                    "Answer like a real human, not an AI. "
                    "Be simple, clear, and personal. "
                    "Use real examples like school projects or coding. "
                    "Keep answers short (4-6 lines max). "
                    "For difficult questions, clearly use STAR method with labels: Situation, Task, Action, Result. "
                    "Do NOT say you are an AI. "
                    "Always include 'because' and 'for example'."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,
            max_tokens=80
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] API failed, using fallback: {e}")
        return fallback_agent(question)


# ---------------------------
# 📊 RUN TASK
# ---------------------------
def run_task(task_name):
    env = InterviewEnv(task_name)
    obs = env.reset()

    total_reward = 0.0
    step_id = 0

    print(f"[START] task={task_name}")

    while True:
        step_id += 1
        question = obs.question

        # 🔥 FORCE STAR FOR HARD TASK
        if task_name == "hard":
            question = f"Answer using STAR method clearly with Situation, Task, Action, Result: {question}"

        # 🧠 Get answer
        answer = llm_agent(question)

        action = Action("respond", answer)
        result = env.step(action)

        reward = float(result.reward)
        total_reward += reward

        print(
            f"[STEP] step={step_id} | question={obs.question} | answer={answer} | reward={reward:.2f}"
        )

        obs = result.observation

        if result.done:
            break

    avg_reward = total_reward / step_id

    print(
        f"[END] task={task_name} | total={total_reward:.2f} | avg={avg_reward:.2f}"
    )

    # Score: 1 (pass) or 0 (fail)
    score = 1 if avg_reward >= 0.5 else 0

    return score


# ---------------------------
# 🚀 MAIN
# ---------------------------
def main():
    tasks = ["easy", "medium", "hard"]

    results = {}

    for task in tasks:
        score = run_task(task)
        results[task] = score

    overall = sum(results.values()) / len(results)

    print("\n[FINAL]")
    for task, score in results.items():
        print(f"{task}: {score:.2f}")

    print(f"overall: {overall:.2f}")


if __name__ == "__main__":
    main()