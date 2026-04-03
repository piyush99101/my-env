from dataclasses import dataclass

@dataclass
class Observation:
    question: str
    step: int

@dataclass
class Action:
    action_type: str   # respond
    content: str

@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool