# import sqlite3
# from collections.abc import Callable
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from contextlib import closing
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from queue import Empty, Queue, SimpleQueue
# from tempfile import TemporaryDirectory
# from time import monotonic
# from typing import Annotated

# from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
# from langchain.agents import create_agent
# from langchain.agents.middleware import wrap_model_call
# from langchain.agents.structured_output import ToolStrategy
# from langchain.messages import HumanMessage, ToolMessage
# from langchain.tools import ToolRuntime, tool
# from pydantic import BaseModel, Field
# from a4ai_project.communication.RouterSystem1 import RouterSystem1

# STOP = object()

# class PlayerInboxes:
#     def __init__(self):
#         self.queues = {}

#     def register(self, name):
#         if name in self.queues:
#             raise ValueError(f"Duplicate player {name}")
#         self.queues[name] = Queue()
#         return self.queues[name]

#     def deliver(self, destination, item, *, exclude=None):
#         names = self.queues if destination == "broadcast" else [destination]
#         for name in names:
#             if name != exclude:
#                 self.queues[name].put(item)

#     def stop(self):
#         self.deliver("broadcast", STOP)

# @dataclass(frozen=True)
# class Entry:
#     id: int
#     author: str
#     body: str

# class Blackboard:
#     def __init__(self, path, router, inboxes):
#         self.path = path
#         self.router = router
#         self.inboxes = inboxes
#         with closing(sqlite3.connect(path)) as db:
#             db.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, author TEXT, body TEXT)")
#             db.commit()

#     def publish(self, author, body, destination, *, exclude=None):
#         with closing(sqlite3.connect(self.path)) as db:
#             cursor = db.execute("INSERT INTO messages(author, body) VALUES (?, ?)", (author, body))
#             entry = Entry(cursor.lastrowid, author, body)
#             db.commit()
#         self.inboxes.deliver(destination, entry, exclude=exclude)
#         return entry

#     def post(self, author, body):
#         destination = self.router.infer_destination(f"{author}: {body}")
#         return self.publish(author, body, destination, exclude=author)

#     def recent(self, limit=10):
#         with closing(sqlite3.connect(self.path)) as db:
#             rows = db.execute(
#                 "SELECT id, author, body FROM messages ORDER BY id DESC LIMIT ?", (limit,)
#             ).fetchall()
#         return [Entry(*row) for row in reversed(rows)]

#     def history(self):
#         with closing(sqlite3.connect(self.path)) as db:
#             return [Entry(*r) for r in db.execute("SELECT id, author, body FROM messages ORDER BY id")]

# @dataclass(frozen=True)
# class PlayerContext:
#     idx: int
#     name: str
#     uuid: int
#     api: DiscoveryWorldAPI
#     board: Blackboard


# class WorldAction(BaseModel):
#     action: str
#     arg1: int | str | None = None
#     arg2: int | str | None = None


# class DialogueChoice(BaseModel):
#     chosen_dialog_option_int: int


# class Wait(BaseModel):
#     """Yield until a board message or private timeout."""
#     reason: str


# class FinishEpisode(BaseModel):
#     """End the whole team's run."""
#     answer: str


# @tool
# def get_observation(runtime: ToolRuntime[PlayerContext]):
#     """Observe your surroundings, world step, and public task progress."""
#     ctx = runtime.context
#     obs = ctx.api.getAgentObservation(agentIdx=ctx.idx)
#     return {k: v for k, v in obs.items() if k != "vision"}


# @tool
# def list_available_actions(runtime: ToolRuntime[PlayerContext]):
#     """List the valid environment actions."""
#     return runtime.context.api.listKnownActions(limited=False)


# @tool
# def list_teleport_locations(runtime: ToolRuntime[PlayerContext]):
#     """List available teleport destinations."""
#     return runtime.context.api.listTeleportLocationsDict()


# @tool
# def post_message(message: str, runtime: ToolRuntime[PlayerContext]):
#     """Publish a message on the shared blackboard."""
#     ctx = runtime.context
#     return f"Posted entry {ctx.board.post(ctx.name, message).id}."


# @tool
# def read_recent_messages(runtime: ToolRuntime[PlayerContext],
#                          limit: Annotated[int, Field(ge=1)] = 10):
#     """Read recent public entries in chronological order."""
#     return [asdict(e) for e in runtime.context.board.recent(limit)]


# @tool
# def tick_environment_world(runtime: ToolRuntime[PlayerContext]):
#     """Advance the shared world one step. The result is private."""
#     return runtime.context.api.tick()


# @tool(return_direct=True)
# def perform_action_in_environment(action: WorldAction | DialogueChoice,
#                                   runtime: ToolRuntime[PlayerContext]):
#     """Attempt one world action. Its result ends this invocation."""
#     ctx = runtime.context
#     result = ctx.api.performAgentAction(agentIdx=ctx.idx, actionJSON=action.model_dump())
#     if result.get("success") is False:
#         result = {**result, "message": ctx.api.ui[ctx.idx].lastActionMessage}
#     return result


# @wrap_model_call
# def one_tool_at_a_time(request, handler):
#     settings = {**request.model_settings, "parallel_tool_calls": False}
#     return handler(request.override(model_settings=settings))


# class Player:
#     def __init__(self, model, context, max_num_steps):
#         self.context = context
#         self.max_num_steps = max_num_steps
#         self.messages = []
#         self.graph = create_agent(
#             model=model,
#             context_schema=PlayerContext,
#             tools=[get_observation, list_available_actions, list_teleport_locations,
#                    post_message, read_recent_messages, tick_environment_world,
#                    perform_action_in_environment],
#             middleware=[one_tool_at_a_time],
#             response_format=ToolStrategy(Wait | FinishEpisode),
#             system_prompt=self._get_system_prompt(),
#         )

#     def _get_system_prompt(self):
#         ctx = self.context
#         return (
#             f"You are {ctx.name}, DiscoveryWorld agent {ctx.idx} (UUID {ctx.uuid}). "
#             "Work toward the scenario task on the blackboard.\n\n"
#             "Use your tools to observe and interact with the world. Each agent gets "
#             "one action attempt per world step. Any agent can tick the shared world. "
#             "You decide when to act, tick, or communicate.\n\n"
#             "Your observations and action results are private. Post to the blackboard "
#             "when you want teammates to know something.\n\n"
#             "Return Wait to resume on a message or timeout. Return FinishEpisode "
#             "with your findings when your observation shows the task has ended "
#             f"or world_steps has reached {self.max_num_steps}."
#         )

#     def invoke(self, wakeup):
#         result = self.graph.invoke(
#             {"messages": [*self.messages, HumanMessage(content=wakeup)]},
#             context=self.context,
#         )
#         self.messages = result["messages"]
#         decision = result.get("structured_response")
#         if isinstance(decision, (Wait, FinishEpisode)):
#             return decision
#         last = self.messages[-1]
#         if isinstance(last, ToolMessage) and last.name == perform_action_in_environment.name:
#             return last
#         raise RuntimeError(f"Unexpected agent exit: {last!r}")


# @dataclass
# class AgentHarness:
#     player: Player
#     inbox: Queue
#     timeout_seconds: float
#     on_finish: Callable
#     trace: list = field(default_factory=list)

#     def run(self):
#         while True:
#             try:
#                 entry = self.inbox.get(timeout=self.timeout_seconds)
#             except Empty:
#                 if self.wake("timeout", f"No inbox message arrived for {self.timeout_seconds:g} seconds."):
#                     return
#                 continue

#             try:
#                 if entry is STOP:
#                     return
#                 if self.wake("message", f"Board #{entry.id} from {entry.author}: {entry.body}", entry.id):
#                     return
#             finally:
#                 self.inbox.task_done()

#     def wake(self, trigger, text, entry_id=None):
#         outcome = self.player.invoke(text)
#         if isinstance(outcome, ToolMessage):
#             kind, details = "action_tool_result", outcome.content
#         else:
#             kind, details = type(outcome).__name__, outcome.model_dump()
#         self.trace.append({"trigger": trigger, "entry_id": entry_id,
#                            "outcome": kind, "details": details})
#         if isinstance(outcome, FinishEpisode):
#             self.on_finish(self.player.context.name, outcome.answer)
#             return True
#         return False


# def run_episode(api, model, roster, task, *, timeout_seconds=30, max_num_steps=50, router=None):
#     """roster maps each nickname to its preconfigured (agent_idx, uuid)."""
#     if not roster or "broadcast" in roster:
#         raise ValueError("Supply players with distinct names other than 'broadcast'.")
#     if len({idx for idx, _ in roster.values()}) != len(roster):
#         raise ValueError("Each player needs its own agent index.")
#     if timeout_seconds <= 0 or max_num_steps <= 0:
#         raise ValueError("Timeout and step ceiling must be positive.")

#     inboxes = PlayerInboxes()
#     finishes = SimpleQueue()
#     harnesses = {}
#     router = RouterSystem1() if router is None else router
#     started = monotonic()

#     def finish(name, answer):
#         finishes.put((name, answer))
#         inboxes.stop()

#     with TemporaryDirectory() as tmp:
#         board = Blackboard(Path(tmp) / "board.sqlite", router, inboxes)
#         for name, (idx, uuid) in roster.items():
#             inbox = inboxes.register(name)
#             router.criteria[name] = f"Messages for {name}, index {idx}, UUID {uuid}"
#             ctx = PlayerContext(idx, name, uuid, api, board)
#             player = Player(model, ctx, max_num_steps)
#             harnesses[name] = AgentHarness(player, inbox, timeout_seconds, finish)

#         board.publish("scenario", task, "broadcast")
#         with ThreadPoolExecutor(max_workers=len(harnesses)) as pool:
#             try:
#                 futures = [pool.submit(harness.run) for harness in harnesses.values()]
#                 for future in as_completed(futures):
#                     future.result()
#             except BaseException:
#                 inboxes.stop()
#                 raise

#         name, answer = finishes.get_nowait()
#         return {
#             "finished_by": name,
#             "answer": answer,
#             "elapsed_seconds": monotonic() - started,
#             "blackboard": [asdict(e) for e in board.history()],
#             "traces": {n: h.trace for n, h in harnesses.items()},
#             "messages": {n: h.player.messages for n, h in harnesses.items()},
#         }
