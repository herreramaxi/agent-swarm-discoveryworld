from langchain.tools import ToolRuntime, tool

from a4ai_project.agents.agentic.context import BlackboardAgentContext

@tool
def post_message_in_blackboard(message: str,runtime: ToolRuntime[BlackboardAgentContext],):
    """Post a message to the shared blackboard under your nickname."""
    ctx = runtime.context
    message_id = ctx.blackboard.post_message(ctx.agent_nickname, message)
    
    return f"Message with message_id '{message_id}' was successfully posted into blackboard"

#TODO: need to think this part, maybe agent will send the parameter last row id
@tool    
def read_messages_from_blackboard(runtime: ToolRuntime[BlackboardAgentContext]):
    """Read blackboard messages after your last read message ID."""
    ctx = runtime.context
    messages = ctx.blackboard.read_messages(last_row_id_read=ctx.last_read_id)
    
    if messages:
        ctx.last_read_id = messages[-1].id
    else:
        return "No messages available after id: {ctx.last_read_id}"
                
    return messages
    
@tool
async def wait_for_blackboard_message(
    runtime: ToolRuntime[BlackboardAgentContext],
    timeout_seconds: int = 20,
) -> str:
    """Wait for a blackboard notification or until the timeout expires."""
    ctx = runtime.context
    notified = await ctx.blackboard.wait(ctx.agent_idx, timeout_seconds)
    return "Notification received." if notified else "Wait timed out."    
