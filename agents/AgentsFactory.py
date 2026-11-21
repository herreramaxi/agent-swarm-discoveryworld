from dataclasses import dataclass
from enum import Enum

from agents.baselines.base import BaseAgent
from agents.baselines.RandomAgent import RandomAgent
from agents.baselines.SingleLLMAgent import SingleLLMAgent
from agents.baselines.SingleLLMAgentDelta import SingleLLMAgentDelta

class AgentType(str, Enum):
    Random = "Random"
    LLMAgent = "LLMAgent"
    LLMAgentDelta = "LLMAgentDelta"
    LLMAgentCompact = "LLMAgentCompact"
    LLMAgentMultiTurn = "LLMAgentMultiTurn"

@dataclass(frozen=True)
class AgentConfiguration:
    agent_type: AgentType


class AgentsFactory:
    @staticmethod
    def create_agent(
        agent_type: AgentType,
        agent_idx: int = 0,
        seed: int | None = None,
        task_description: str = None,
        possible_actions: dict = None,
        teleport_locations :dict = None
    ) -> BaseAgent:
        if agent_type is AgentType.Random:
            return RandomAgent(agent_idx=agent_idx,  
                               possible_actions=possible_actions,
                               teleport_locations=teleport_locations,
                               seed=seed)
        if agent_type in (AgentType.LLMAgent, AgentType.LLMAgentCompact, AgentType.LLMAgentMultiTurn):
            return SingleLLMAgent(
                agent_idx=agent_idx,
                task_description=task_description,
                possible_actions=possible_actions,
                teleport_locations=teleport_locations,
                compact_prompt=agent_type is not AgentType.LLMAgent,
                # multi_turn_prompt=agent_type is AgentType.LLMAgentMultiTurn,
            )
        if agent_type is AgentType.LLMAgentDelta:
            return SingleLLMAgentDelta(task_description = task_description, possible_actions= possible_actions, teleport_locations= teleport_locations,agent_idx=agent_idx)        
        raise ValueError(f"Unsupported agent type: {agent_type!r}")
