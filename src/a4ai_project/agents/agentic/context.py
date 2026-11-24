
from dataclasses import dataclass
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

from a4ai_project.communication.blackboard import Blackboard

@dataclass        
class AgentContext:
    agent_idx: int
    agent_nickname: str
    environment: DiscoveryWorldAPI
      
@dataclass        
class BlackboardAgentContext(AgentContext):
    blackboard: Blackboard     
    last_read_id: int = 0
