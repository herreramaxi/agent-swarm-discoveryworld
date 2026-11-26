import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from queue import Empty, Queue
from time import perf_counter
from typing import Any
from discoveryworld import DiscoveryWorldAPI
from langsmith import BaseModel
from a4ai_project.agents.agentic.BlackboardAgenticPlayer import BlackboardAgenticPlayer
from a4ai_project.agents.shared_model import FinishEpisode
from a4ai_project.communication.blackboard import Blackboard, BoardMessage

STOP = object()  # The runner imports this same object.

@dataclass
class AgentHarness:    
    agent_idx: int
    agent_nickname: str
    model: BaseModel
    environment: DiscoveryWorldAPI     
    max_num_steps: int 
    agent: BlackboardAgenticPlayer | None = None
    blackboard: Blackboard|None = None
    uuid: int = 0
    queue: Queue | None = None
    invocation_trace: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        self.agent = BlackboardAgenticPlayer(agent_idx = self.agent_idx,
                                        uuid=self.uuid,
                                        agent_nickname= self.agent_nickname,
                                        model = self.model,
                                        environment=self.environment,
                                        max_num_steps=self.max_num_steps,
                                        blackboard = self.blackboard                                                                                 
                                        )
        self.queue = self.blackboard.subscribe(self.agent)        
    
    def invoke(self) -> dict | None:
        timeout=30
        while True:
            try:
                message = self.queue.get(timeout=timeout)
            except Empty:
                response, finished = self.invoke_agent(f"No blackboard message arrived in the last {timeout} seconds. Decide what to do next.")
                if finished:
                    return response
                continue

            try:
                if message is STOP:
                    return None

                if not isinstance(message, BoardMessage):
                    raise TypeError(f"Unexpected inbox item: {type(message).__name__}")

                response, finished = self.invoke_agent(
                    f"Blackboard message received => author: {message.author}, body: {message.body}, created_at: {message.created_at}",
                    message_id=message.id,
                )
                if finished:
                    return response
            finally:
                self.queue.task_done()
                     
    def invoke_agent(self, message: str, *, message_id: int | None = None):
        started_at = datetime.now(UTC)
        start = perf_counter()
        outcome = "error"
        try:
            response = self.agent.invoke(message)
            structured_response = response["structured_response"]
            outcome = type(structured_response).__name__
        finally:
            record = {
                "agent": self.agent_nickname,
                "trigger": "timeout" if message_id is None else "message",
                "message_id": message_id,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(UTC).isoformat(),
                "duration_seconds": perf_counter() - start,
                "outcome": outcome,
            }
            self.invocation_trace.append(record)
            print(f"Harness invocation: {json.dumps(record)}", flush=True)

        print(f"invoke_agent: {response}")
        return response, isinstance(structured_response, FinishEpisode)
        
