from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.baselines.base import BaseAgent, StepView, active_dialog

_NAVIGATION_ACTIONS = {
    "MOVE_DIRECTION",
    "ROTATE_DIRECTION",
    "TELEPORT_TO_OBJECT",
    "TELEPORT_TO_LOCATION",
}

class ActionResponse(BaseModel):
    """An action and its arguments for the DiscoveryWorld agent."""

    action: str = Field(description="Name of the action to perform")
    arg1: int | str | None = Field(description="First argument, or null if unused")
    arg2: int | str | None = Field(description="Second argument, or null if unused")


class DialogResponse(BaseModel):
    """An option selected from the current dialog."""

    chosen_dialog_option_int: int = Field(description="Number of the dialog option to select")


#TODO: Need to remove this I think
def navigation_loop_detected(actions_taken, window: int = 6) -> bool:
    """Detect repeated positions during travel with unchanged inventory."""
    recent = actions_taken[-window:]
    if len(recent) < window or any(record["action"].get("action") not in _NAVIGATION_ACTIONS for record in recent):
        return False
    inventories = [json.dumps(record.get("inventory", []), sort_keys=True) for record in recent]
    positions = [(record["location"]["x"], record["location"]["y"]) for record in recent]
    return len(set(inventories)) == 1 and len(set(positions)) <= window // 2

from llm import get_llm
@dataclass
class SingleLLMAgent(BaseAgent):
    """Chooses one action with an LLM. The runner submits it and ticks the world."""
    model: BaseModel
    task_description: str = ""
    possible_actions: dict = field(default_factory=dict)
    teleport_locations: dict = field(default_factory=dict)  
    compact_prompt: bool = False
    
    def choose_action(self, view: StepView) -> dict:
        in_dialog = active_dialog(view.observation) is not None
        prompt = self._prompt(view, in_dialog=in_dialog)
        schema = DialogResponse if in_dialog else ActionResponse
        response = self.model.with_structured_output(schema).invoke(prompt)
        return response.model_dump(mode="json")

    def _prompt(self, view: StepView, in_dialog: bool) -> str:
        json_options = {"separators": (",", ":")} if self.compact_prompt else {"indent": 2}
        prompt = (
            f"You are DiscoveryWorld agent {self.agent_idx}."
            f"Your task is: {self.task_description}\n"          
            "Choose one action at a time and check its outcome before continuing.\n"
            "Teleporting moves only you, not the target object. Pick up movable objects "
            "before taking them elsewhere. Both arguments of USE must be accessible "
            "simultaneously (in inventoryObjects or accessibleEnvironmentObjects).\n"
            "After reaching an object, perform the needed interaction before leaving. "
            "Use directions_you_can_move for movement; rotate to face nearby objects.\n"
            "Review your history. Avoid repeating travel without task progress. "
            "If travel repeats, identify the missing interaction or prerequisite.\n"
            "For chemical experiments, keep track of chemicals tried and outcomes; "
            "follow the task's instructions for cleaning the jar between trials.\n"
            "Use only supplied UUIDs and valid actions. Use null for unused arguments.\n"
        )
        prompt += "Current observation:\n" + json.dumps(view.observation["ui"], **json_options)
        prompt += "\nAvailable actions:\n" + json.dumps(self.possible_actions, **json_options)
        prompt += "\nTeleport locations:\n" + json.dumps(self.teleport_locations, **json_options)
        prompt += "\nAction history (including outcomes):\n" + json.dumps(view.history, **json_options)
        # if navigation_loop_detected(view.history):
        #     print(f"Agent {self.agent_idx}: repeated travel detected.")
        #     prompt += (
        #         "\nLOOP DETECTED: Recent navigation revisits the same positions with "
        #         "unchanged inventory. Identify the missing prerequisite. Prefer an "
        #         "accessible interaction that advances the task; collect needed movable "
        #         "objects before leaving. Do not repeat the same travel cycle."
        #     )
        if in_dialog:
            prompt += (
                "\nYou are in dialog. Select a listed option from dialog_box and return "
                "chosen_dialog_option_int instead of an ordinary action."
            )
        return prompt
