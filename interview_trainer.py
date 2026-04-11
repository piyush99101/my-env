import random
import time
import re
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("HF_TOKEN"))

# 🎯 Role-based questions
QUESTION_BANK = {
    "developer": [
        "Tell me about yourself.",
        "Explain a project you have built.",
        "How do you debug code?",
        "Describe a technical challenge you solved."
    ],
    "designer": [
        "Tell me about yourself.",
        "Explain your design process.",
        "How do you handle feedback?",
        "Describe a creative project."
    ],
    "general": [
        "Tell me about yourself.",
        "What are your strengths?",
        "Why should we hire you?",
        "Describe a challenge you faced."
    ]
}

# ---------------------------
# 🤖 AI THINKING (Groq)
# ---------------------------
def ai_analysis(answer, question):
    prompt = f"""
You are an expert interview evaluator.

Question: {question}
Answer: {answer}

Analyze the answer deeply:
- Is it relevant?
- Does it show understanding?
- Is it strong or weak?

Give:
1. A short evaluation
2. 1 improvement suggestion
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ---------------------------
# 🧠 RULE-BASED THINKING
# ---------------------------
def evaluate_answer(answer, response_time):
    answer_lower = answer.lower()
    words = answer_lower.split()

    score = 0.0
    feedback = []

    # Depth
    if len(words) > 40:
        score += 0.3
        feedback.append("✔ Detailed")
    elif len(words) > 20:
        score += 0.2
        feedback.append("✔ Good length")
    else:
        score += 0.1
        feedback.append("❌ Too short")

    # Structure
    if len(re.split(r'[.!?]+', answer)) >= 3:
        score += 0.2
        feedback.append("✔ Structured")
    else:
        feedback.append("❌ Poor structure")

    # Reasoning
    if any(w in answer_lower for w in ["because", "therefore", "so"]):
        score += 0.2
        feedback.append("✔ Reasoning present")

    # Example
    if "for example" in answer_lower:
        score += 0.2
        feedback.append("✔ Example used")

    # Confidence
    if any(w in answer_lower for w in ["achieved", "built", "led"]):
        score += 0.2
        feedback.append("✔ Confident tone")

    # Time factor
    if response_time < 5:
        score += 0.1
        feedback.append("✔ Fast response")
    elif response_time > 20:
        score -= 0.1
        feedback.append("⚠ Too slow")

    final_score = max(0.01, min(0.99, score))
    
    # Add more detailed feedback
    if final_score < 0.5:
        feedback.append("❌ Your answer needs more depth, structure, and reasoning.")
    elif final_score < 0.7:
        feedback.append("⚠ Your answer is good, but could be improved with more detail and clarity.")
    else:
        feedback.append("✔ Great job! Your answer demonstrates strong depth, structure, and reasoning.")

    return final_score, feedback


# ---------------------------
# 🚀 MAIN SYSTEM
# ---------------------------
def run_interview():
    print("\n🎤 AI Interview Trainer (AI + Thinking Model)\n")

    # User info
    name = input("Enter your name: ")
    qualification = input("Enter your qualification: ")
    role = input("Role (developer/designer/general): ").lower()

    questions = QUESTION_BANK.get(role, QUESTION_BANK["general"])
    questions = random.sample(questions, 3)

    total_score = 0

    for i, q in enumerate(questions):
        print(f"\nQ{i+1}: {q}")

        start = time.time()
        answer = input("Your Answer: ")
        end = time.time()

        response_time = end - start

        # 🧠 Rule-based score
        score, feedback = evaluate_answer(answer, response_time)

        # 🤖 AI analysis
        ai_feedback = ai_analysis(answer, q)

        print("\n📊 Score:", round(score * 10, 1), "/10")
        print("⏱ Response Time:", round(response_time, 2), "sec")

        print("\n🧠 System Feedback:")
        for f in feedback:
            print("-", f)

        print("\n🤖 AI Feedback:")
        print(ai_feedback)

        total_score += score

    avg = total_score / 3

    print("\n🏁 FINAL RESULT")
    print(f"Average Score: {avg:.2f}/10")

    if avg >= 8:
        print("🏆 Excellent performance!")
    elif avg >= 5:
        print("👍 Good job!")
    else:
        print("⚠ Needs improvement")


if __name__ == "__main__":
    # Quick local test
    test_answer = "This is a detailed answer with good structure, reasoning, and examples. I have achieved many things in my career."
    test_score, test_feedback = evaluate_answer(test_answer, 10)
    print(f"[TEST] Score: {test_score:.2f} / 1.0")
    for item in test_feedback:
        print("- ", item)
    print()

    run_interview()
