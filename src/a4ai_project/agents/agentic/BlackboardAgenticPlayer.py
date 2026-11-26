from a4ai_project.agents.agentic.context import AgentContext
from a4ai_project.agents.agentic.tools.blackboard_tools import post_message_in_blackboard, read_recent_messages_from_blackboard
from a4ai_project.agents.agentic.tools.environment_tools import get_observation, list_available_actions, list_teleport_locations_available, perform_action_in_environment, tick_environment_world, tick_environment_world_old
from dataclasses import dataclass, field
import operator
from typing import Annotated, Any, NotRequired
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage
from langsmith import BaseModel
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from langchain.agents.structured_output import ToolStrategy
from a4ai_project.agents.shared_model import ContinueEpisode, FinishEpisode
from a4ai_project.communication.blackboard import Blackboard
    
class PlayerState(AgentState):
    actions_succeeded: NotRequired[Annotated[int, operator.add]]
    actions_failed: NotRequired[Annotated[int, operator.add]]
    completion_reported: NotRequired[Annotated[bool, operator.or_]]
    
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
    actions_succeeded: int = 0
    actions_failed: int = 0
    
    def __post_init__(self):  
        self.environment.ui[self.agent_idx]
        agent_ui = self.environment.ui[self.agent_idx]
        current_Agent = agent_ui.currentAgent
        self.uuid = current_Agent.uuid
                
        self.agent = create_agent(
            model = self.model,
            context_schema = AgentContext,
            state_schema = PlayerState,
            tools = [
                list_available_actions, 
                list_teleport_locations_available, 
                get_observation, 
                perform_action_in_environment, 
                tick_environment_world,
                
                post_message_in_blackboard,
                read_recent_messages_from_blackboard
            ],
            middleware= [],         
            system_prompt= self._get_system_prompt(),
            response_format=ToolStrategy(ContinueEpisode | FinishEpisode)            
        )
    
    def invoke(self, input:str):
        context = AgentContext(
            agent_idx=self.agent_idx,
            agent_nickname=self.agent_nickname,
            environment=self.environment,
            blackboard=self.blackboard,
        )
        human_message = HumanMessage(input)
        self.message_history.append(human_message)

        response = self.agent.invoke(
            {"messages": list(self.message_history)},
            context=context,
        )
        self.message_history = list(response["messages"])
        self.actions_succeeded += int(response.get("actions_succeeded") or 0)
        self.actions_failed += int(response.get("actions_failed") or 0)

        return {
            "structured_response": response["structured_response"],
            "actions_succeeded": self.actions_succeeded,
            "actions_failed": self.actions_failed,
        }
        
    def _get_system_prompt(self) -> str:            
        return f"""You are an autonomous scientific discovery agent in DiscoveryWorld.
Work with your peers to complete the user's scientific task as efficiently
as possible.

## Identity

- Nickname: @{self.agent_nickname}
- Agent index: {self.agent_idx}
- UUID: {self.uuid}

## Hard limits

- **Step limit:** Monitor `world_steps` in your observations. If it reaches
    or exceeds {self.max_num_steps}, stop and return `FinishEpisode`.
- **Action limit:** Call `perform_action_in_environment` at most once per
    wake-up. Each agent may also act at most once per world step. Receiving
    another message does not grant a second action in the same step.

## Wake-up protocol

1. **Read the blackboard.** Before acting, call
    `read_recent_messages_from_blackboard`. Check what peers have discovered,
    which subtasks they have claimed, and who has acted this step.
2. **Prioritize mentions.** Address requests for `@{self.agent_nickname}`
    or `@all` before starting unrelated exploration.
3. **Coordinate with peers.** Use `@nickname` when handing off a turn,
    claiming a subtask, requesting help, or asking a peer to tick. Reserve
    `@all` for discoveries that affect everyone and step-completion broadcasts.
4. **Coordinate the tick.** The world advances only when someone calls
    `tick_environment_world`. Use blackboard mentions to coordinate who
    will tick. Wait until all active peers have finished acting for the
    current world step before advancing it.

## Environment rules

- **Observe:** Call `get_observation` to inspect your surroundings,
    inventory, and `world_steps`. Check the effects of an action after
    the action executes and the world ticks.
- **Handle failures:** Read the error message and take a fresh observation.
    Identify the missing prerequisite before retrying on a later eligible turn.
- **Navigate:** Use `directions_you_can_move` for step-by-step movement.
    Rotate to face nearby objects before interacting. Teleporting moves only
    your agent. Avoid repeating routes that produce no discoveries or progress.
- **Manipulate objects:** Pick up movable objects before transporting them.
    For `USE`, both `arg1` and `arg2` must be accessible at the same time via
    `inventoryObjects` or `accessibleEnvironmentObjects`. Complete the needed
    interactions at an object before leaving it.
- **Format actions:** Use valid action names and supplied UUIDs. Pass
    `null` (`None` in Python) for unused arguments.

## End your turn

- Return `ContinueEpisode` when you have finished acting or communicating
    for this wake-up and are ready to wait for the next blackboard update
    or world step.
- Return `FinishEpisode` only when the overall scientific task is solved
    or `world_steps` has reached {self.max_num_steps}.
    """
        
    def _get_system_prompt2(self):
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
