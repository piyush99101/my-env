from env.models import Observation, StepResult
from env.tasks import TASKS

class InterviewEnv:
    def __init__(self, task_name):
        self.task_name = task_name
        self.questions = TASKS[task_name]
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return Observation(
            question=self.questions[0],
            step=0
        )

    def evaluate(self, answer):
        answer = answer.lower()
        score = 0.0

        # Depth
        if len(answer.split()) > 10:
            score += 0.3

        # Reasoning
        if "because" in answer or "so" in answer:
            score += 0.3

        # Example
        if "example" in answer or "for instance" in answer:
            score += 0.4

        # Clamp to strictly open interval (0, 1) as required by the platform
        score = max(0.01, min(score, 0.99))
        return score

    def step(self, action):
        reward = self.evaluate(action.content)

        self.current_step += 1
        done = self.current_step >= len(self.questions)

        next_q = "" if done else self.questions[self.current_step]

        return StepResult(
            observation=Observation(
                question=next_q,
                step=self.current_step
            ),
            reward=reward,
            done=done
        )