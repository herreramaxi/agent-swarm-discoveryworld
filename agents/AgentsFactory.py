from dataclasses import dataclass
from enum import Enum

from agents.base import BaseAgent
from agents.RandomAgent import RandomAgent
from agents.SingleLLMAgent import SingleLLMAgent


class AgentType(str, Enum):
    Random = "Random"
    LLMAgent = "LLMAgent"


@dataclass(frozen=True)
class AgentConfiguration:
    agent_type: AgentType


class AgentsFactory:
    @staticmethod
    def create_agent(
        agent_type: AgentType,
        agent_idx: int = 0,
        seed: int | None = None,
    ) -> BaseAgent:
        if agent_type is AgentType.Random:
            return RandomAgent(agent_idx=agent_idx, seed=seed)
        if agent_type is AgentType.LLMAgent:
            return SingleLLMAgent(agent_idx=agent_idx)
        raise ValueError(f"Unsupported agent type: {agent_type!r}")
