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
        score = 0

        # Using small decimal values guarantees that even if ALL or NO 
        # conditions are met, the score mathematically evaluates to 
        # something strictly greater than 0.0 and strictly less than 1.0!
        score = 0.1

        # Depth
        if len(answer.split()) > 10:
            score += 0.3

        # Reasoning
        if "because" in answer or "so" in answer:
            score += 0.3

        # Example
        if "example" in answer or "for instance" in answer:
            score += 0.2

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