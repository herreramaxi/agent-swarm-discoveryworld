from a4ai_project.agents.agentic.AgenticPlayer import AgentContext
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from langsmith import BaseModel
from pydantic import Field

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
    """Attempt one action in the environment.

    On failure, diagnose the returned message and choose a corrected or
    prerequisite action before continuing.

    Args:
        action (ActionResponse | DialogResponse): The action to be performed in the environment.
            The action can be DialogResponse only if agent is within dialogue.
    """    
    context = runtime.context
    agent_idx = context.agent_idx
    env = context.environment
    action_dict = action.model_dump(mode="json")

    result = env.performAgentAction(agentIdx=agent_idx, actionJSON=action_dict)
    if result.get("success") is False:
        result = {**result, "message": env.ui[agent_idx].lastActionMessage}

    succeeded = result.get("success") is True
    result["success"] = succeeded
    print(f"ACTION: {action}")
    print(f"ACTION RESULT: {result}")

    count_key = "actions_succeeded" if succeeded else "actions_failed"
    return Command(update={
        count_key: 1,
        "messages": [
            ToolMessage(content=str(result), tool_call_id=runtime.tool_call_id)
        ],
    })

@tool
def tick_environment_world(runtime: ToolRuntime[AgentContext]):
    """Advance the DiscoveryWorld simulation by one step.

    Call this tool only once per step, after every agent has performed exactly
    one action in that step.
    """
    environment = runtime.context.environment
    response = environment.tick()
    if response["success"]:
        print(f"-----------> STEP: {environment.world.getStepCounter()}")
    else:
        print("-----------> tick_environment_world FAILED")
    return response
    
