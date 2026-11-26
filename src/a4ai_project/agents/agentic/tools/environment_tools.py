from a4ai_project.agents.agentic.context import AgentContext, AgentContext
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from langsmith import BaseModel
from pydantic import Field

from a4ai_project.agents.shared_model import DialogueChoice, WorldAction


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
    """Observe your surroundings and public task progress.
    Pay attention to flag is_in_dialog. If the agent is in dialog means a dialogue action must be performed.
    """
    
    agent_idx = runtime.context.agent_idx
    env = runtime.context.environment
    
    obs = env.getAgentObservation(agentIdx=agent_idx)
    obs.pop("vision", None)
    # print(obs)
    return obs
    
#TODO: maybe we should limit or cap to 1  valid action this tool via middleware
@tool
def perform_action_in_environment(action: WorldAction| DialogueChoice , runtime: ToolRuntime[AgentContext]):
    """Attempt one action in the environment.

    On failure, diagnose the returned message and choose a corrected or
    prerequisite action before continuing.

    Notify other agents in the environment via blackboard.
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

#TODO: we need to figure it out if we need to broadcast events manually or just let the LLM to post it via tools
@tool
def tick_environment_world(runtime: ToolRuntime[AgentContext]):
    """Advances the world simulation by 1 step
    
    Important:
    - This tool requires coordination between agents, use blackboard for that.
    - Only call this once all active agents have submitted their action for the current step.
    - Once this tools is executed, notify other agents in the environment via blackboard."""
    
    ctx =  runtime.context
    environment = ctx.environment
    response = environment.tick()
    if response["success"]:
        step = environment.world.getStepCounter()
        print(f"-----------> STEP: {environment.world.getStepCounter()}")
        # ctx.blackboard.broadcast_message(author= ctx.agent_nickname, 
        #                                  message = f"I have advanced the world to step {step}.",   
        #                                  sender_idx= ctx.agent_idx
        #                                  )
    return { **response, "world_step": step }

# @tool
# def report_task_completion(runtime: ToolRuntime[AgentContext]):
#     """Tell the runner that the scenario's tasks have ended."""
     
#     #we need to double check that the task is completed or not
#     ctx = runtime.context
#     obs = ctx.environment.getAgentObservation(ctx.agent_idx)
    
#     if obs.get("errors"):
#         return f"Could not verify completion: {obs['errors']}"
    
#     #TODO: extract into task?
#     if not ctx.environment.areTasksComplete():
#         return "The task is not completed yet, pending items are still unfinished"   
    
@tool
def tick_environment_world_old(runtime: ToolRuntime[AgentContext]):
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
        
