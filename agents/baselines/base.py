from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepView:
    """What an agent may read when choosing one action.

    The runner builds this. Agents do not submit the action or advance the world.
    """
    observation: dict 
    history: list


def active_dialog(observation: dict) -> dict | None:
    """Return the dialog box when this agent must choose a dialog line."""
    box = observation.get("ui", {}).get("dialog_box") or {}
    if box.get("is_in_dialog") and box.get("dialogOptions"):
        return box
    return None


@dataclass
class BaseAgent(ABC):
    """A policy that returns one action dict for the current view."""

    agent_idx: int = 0

    @abstractmethod
    def choose_action(self, view: StepView) -> dict:
        """Return the next action. Do not call performAgentAction or tick."""
