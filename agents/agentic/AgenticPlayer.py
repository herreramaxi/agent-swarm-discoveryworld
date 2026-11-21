from dataclasses import dataclass, field
from typing import Any
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langsmith import BaseModel
from pydantic import Field
from environment.discoveryworld.discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

@dataclass        
class AgentContext:
    agent_idx: int
    environment: DiscoveryWorldAPI
            
@dataclass
class AgenticPlayer:
    agent_idx: int
    model: BaseModel
    environment: DiscoveryWorldAPI    
    agent: Any | None = None
    message_history: list[Any] = field(default_factory=list)
    
    def __post_init__(self):  
        self.agent = create_agent(
            model = self.model,
            context_schema = AgentContext,
            tools = [
                list_available_actions, 
                list_teleport_locations_available, 
                get_observation, 
                perform_action_in_environment, 
                tick_environment_world
            ],
            system_prompt= self._get_system_prompt(),
            middleware= []            
        )
    
    def invoke(self, input:str):
        context = AgentContext(self.agent_idx,self.environment)
        human_message = HumanMessage(input)
        self.message_history.append(human_message)
        
        response = self.agent.invoke({"messages": self.message_history}, context=context)
        return response
        
    def _get_system_prompt(self):
        prompt = f"""You are DiscoveryWorld agent, you are identiied by agent_idx: {self.agent_idx}.
        You must accomplish the user task in the most efficiently way possible. Use the tools available to interact with the environment.
        Recommendations:
        - You can get observations to verify the current status of the environment, which is only updated after performing actions on it.
        - Choose one action at a time and check its outcome before continuing.
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

@tool
def list_available_actions(runtime: ToolRuntime[AgentContext]):
    """List all the available actions an agent can perform in a given enviroment. 
    These actions would not change once the environment is created.
    """
    return runtime.context.environment.listKnownActions(limited=False)

@tool
def list_teleport_locations_available(runtime: ToolRuntime[AgentContext]):
    """List all the available teleport locations an agent can use to teleport within a given enviroment. 
    These teleport locations would not change once the environment is created.
    """
    return runtime.context.environment.listTeleportLocationsDict()

@tool
def get_observation(runtime: ToolRuntime[AgentContext]):
    """Get an observation from the environment. 
    Pay attention to flag is_in_dialog. If the agent is in dialog means a dialogue action must be performed.
    """
    
    agent_idx = runtime.context.agent_idx
    env = runtime.context.environment
    
    obs = env.getAgentObservation(agentIdx=agent_idx)
    obs.pop("vision", None)
    # print(obs)
    return obs

class ActionResponse(BaseModel):
    """An action and its arguments for the DiscoveryWorld agent (Use this when agent is not in dialogue)."""
    action: str = Field(description="Name of the action to perform, or null if ")
    arg1: int | str | None = Field(description="First argument, or null if unused")
    arg2: int | str | None = Field(description="Second argument, or null if unused")
    # chosen_dialog_option_int: int = Field(description="Number of the dialog option to select")
    
class DialogResponse(BaseModel):
    """An option selected from the current dialog (Use this when agent is in dialogue)."""
    chosen_dialog_option_int: int = Field(description="Number of the dialog option to select")

@tool
def perform_action_in_environment(action: ActionResponse | DialogResponse, runtime: ToolRuntime[AgentContext]):
    """Perform an action in the environment. 
    After an action is performed, the 'tick_environment_world' tool need to be called.
    
    Args:
        action (ActionResponse | DialogResponse): The action to be performed in the environment.
            The action can be DialogResponse only if agent is within dialogue.
    """    
    agent_idx = runtime.context.agent_idx
    env = runtime.context.environment
    action_dict = action.model_dump(mode="json")
    
    result = env.performAgentAction(agentIdx=agent_idx, actionJSON=action_dict)
    return result

@tool()
def tick_environment_world(runtime: ToolRuntime[AgentContext]):
    """Tick the environment (world). 
    This need to be done after all the agents have performed their actions (one per agent) within a step.
    This tool cannot be called multiple times within a step.
    """
    environment = runtime.context.environment
    response = environment.tick()
    if response["success"]:
        print(f"-----------> STEP: {environment.world.getStepCounter()}")
    else: 
        print(f"-----------> tick_environment_world FAILED")
    return response
    
