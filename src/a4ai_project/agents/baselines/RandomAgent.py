from dataclasses import dataclass, field
import random
import time

from a4ai_project.agents.baselines.base import BaseAgent, StepView, active_dialog

_DIRECTIONS = ("north", "east", "south", "west")
_SPECIAL_ACTIONS = (
    "MOVE_DIRECTION",
    "ROTATE_DIRECTION",
    "TELEPORT_TO_LOCATION",
    "DISCOVERY_FEED_GET_POST_BY_ID",
)


@dataclass
class RandomAgent(BaseAgent):
    """Picks one action at random. The runner submits it and ticks the world."""
    
    possible_actions: dict = field(default_factory=dict)
    teleport_locations: dict = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random()
        if self.seed is None:
            int_time = int(time.time())
            self._rng.seed(int_time)
            print("Random Agent Seed: " + str(int_time))
        else:
            self._rng.seed(self.seed)
            print("Random Agent Seed: " + str(self.seed))

    def choose_action(self, view: StepView) -> dict:
        dialog = active_dialog(view.observation)
        if dialog is not None:
            return self._dialog_action(dialog)
        return self._world_action(view)

    def _dialog_action(self, dialog: dict) -> dict:
        possible_options = list(dialog["dialogOptions"].keys())
        chosen_option = self._rng.choice(possible_options)
        try:
            chosen_option_int = int(chosen_option)
        except (TypeError, ValueError):
            chosen_option_int = self._rng.randint(1, len(possible_options) + 1)
        return {"chosen_dialog_option_int": chosen_option_int}

    def _world_action(self, view: StepView) -> dict:
        accessible_objects = (
            view.observation["ui"]["inventoryObjects"]
            + view.observation["ui"]["accessibleEnvironmentObjects"]
        )
        action_name = self._rng.choice(list(self.possible_actions.keys()))
        action_args = self.possible_actions[action_name]["args"]
        action = {
            "action": action_name,
            "arg1": None,
            "arg2": None,
        }

        if action_args and not accessible_objects and action_name not in _SPECIAL_ACTIONS:
            print("No accessible objects for " + action_name + ". Moving instead.")
            action_name = "MOVE_DIRECTION"
            action["action"] = action_name
            action_args = []

        if action_name not in _SPECIAL_ACTIONS:
            for arg in action_args:
                action[arg] = self._rng.choice(accessible_objects)["uuid"]

        if action_name == "MOVE_DIRECTION":
            action["arg1"] = self._rng.choice(_DIRECTIONS)
        elif action_name == "ROTATE_DIRECTION":
            action["arg1"] = self._rng.choice(_DIRECTIONS)
        elif action_name == "TELEPORT_TO_LOCATION":
            teleport_locations = list(self.teleport_locations.keys())
            if not teleport_locations:
                print("No teleport locations. Moving instead.")
                action["action"] = "MOVE_DIRECTION"
                action["arg1"] = self._rng.choice(_DIRECTIONS)
            else:
                action["arg1"] = self._rng.choice(teleport_locations)
        elif action_name == "DISCOVERY_FEED_GET_POST_BY_ID":
            action["arg1"] = self._rng.randint(1, 100)

        return action
