from langchain.tools import ToolRuntime, tool

from a4ai_project.agents.agentic.context import AgentContext

@tool
def post_message_in_blackboard(message: str,runtime: ToolRuntime[AgentContext],):
    """Publish a message on the shared blackboard."""
    ctx = runtime.context
    posted_message = ctx.blackboard.post_message(ctx.agent_nickname, message)
    
    return f"Posted message into blackboard -> {posted_message.get_formatted_messsage()}"

@tool    
def read_recent_messages_from_blackboard(runtime: ToolRuntime[AgentContext], limit:int =10):
    """Read blackboard messages after your last read message ID."""
    ctx = runtime.context
    messages = ctx.blackboard.read_messages(limit=limit)
    
    if not messages:        
        return "No messages available: []"
                
    return messages
    
# @tool
# async def wait_for_blackboard_message(
#     runtime: ToolRuntime[AgentContext],
#     timeout_seconds: int = 20,
# ) -> str:
#     """Wait for a blackboard notification or until the timeout expires."""
#     ctx = runtime.context
        
#     notified = await ctx.blackboard.wait(ctx.agent_idx, timeout_seconds)
#     return "Notification received." if notified else "Wait timed out."    
