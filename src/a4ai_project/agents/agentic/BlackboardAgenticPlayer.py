from a4ai_project.agents.agentic.context import BlackboardAgentContext
from a4ai_project.agents.agentic.tools.blackboard_tools import post_message_in_blackboard, read_messages_from_blackboard, wait_for_blackboard_message
from a4ai_project.agents.agentic.tools.environment_tools import get_observation, list_available_actions, list_teleport_locations_available, perform_action_in_environment, tick_environment_world
from dataclasses import dataclass, field
import operator
from typing import Annotated, Any, NotRequired
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage
from langsmith import BaseModel
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

from a4ai_project.communication.blackboard import Blackboard


    
class PlayerState(AgentState):
    actions_succeeded: NotRequired[Annotated[int, operator.add]]
    actions_failed: NotRequired[Annotated[int, operator.add]]
    
@dataclass
class BlackboardAgenticPlayer:
    agent_idx: int
    agent_nickname: str
    model: BaseModel
    environment: DiscoveryWorldAPI   
    blackboard: Blackboard
    max_num_steps: int 
    agent: Any | None = None
    message_history: list[Any] = field(default_factory=list)
    uuid: int = 0
    
    def __post_init__(self):  
        self.environment.ui[self.agent_idx]
        agent_ui = self.environment.ui[self.agent_idx]
        current_Agent = agent_ui.currentAgent
        self.uuid = current_Agent.uuid
        
        # self.blackboard.sys
        
        self.agent = create_agent(
            model = self.model,
            context_schema = BlackboardAgentContext,
            state_schema = PlayerState,
            tools = [
                list_available_actions, 
                list_teleport_locations_available, 
                get_observation, 
                perform_action_in_environment, 
                tick_environment_world,
                
                post_message_in_blackboard,
                read_messages_from_blackboard,
                wait_for_blackboard_message
            ],
            system_prompt= self._get_system_prompt(),
            middleware= []            
        )
    
    def invoke(self, input:str):
        context = BlackboardAgentContext(self.agent_idx, self.environment, self.blackboard)
        human_message = HumanMessage(input)
        self.message_history.append(human_message)
        
        response = self.agent.invoke({"messages": self.message_history}, context=context)
        
        return response
        
    def _get_system_prompt(self):
        prompt = f"""You are an agent from DiscoveryWorld environment. 
        You are identiied as agent_idx {self.agent_idx}, with internal uuid {self.uuid} and nickname {self.agent_nickname}. 
        
        Directives: 
        - You must accomplish the user task in the most efficiently way possible. 
        - You cannot continue running if you have reached the maximum number of steps '{self.max_num_steps}' in the world (check world_steps from observations).         
        - Use the tools available to interact with the environment.        
        
        Recommendations:
        - Prefer the use of nicknames to communicate with other agents.
        - You can get observations to verify the current status of the environment, which is only updated after performing actions on it.
        - Choose one action at a time and check its outcome before continuing.
        - If an action fails, use its error or message and get a fresh observation when needed. Identify a corrected or prerequisite action before continuing.
        - After an action succeeds, call tick_environment_world before attempting another action.
        - Teleporting moves only you, not the target object. 
        - Pick up movable objects before taking them elsewhere. 
        - Both arguments of USE must be accessible simultaneously (in inventoryObjects or accessibleEnvironmentObjects).
        - After reaching an object, perform the needed interaction before leaving. 
        - Use directions_you_can_move for movement; rotate to face nearby objects.
        - Review your history. Avoid repeating travel without task progress. 
        - If travel repeats, identify the missing interaction or prerequisite.
        - Use only supplied UUIDs and valid actions. Use null for unused arguments.      
        """
                
        return prompt