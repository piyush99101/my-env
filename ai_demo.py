import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_answer(question, answer):
    prompt = f"""
You are an expert interviewer.

Question: {question}
Answer: {answer}

Evaluate quality and give improvement.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    q = input("Question: ")
    a = input("Answer: ")

    print("\nAI Feedback:")
    print(analyze_answer(q, a))