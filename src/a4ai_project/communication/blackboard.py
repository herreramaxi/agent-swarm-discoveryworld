import asyncio
from collections import defaultdict
from a4ai_project.agents.agentic.BlackboardAgenticPlayer import BlackboardAgenticPlayer
from a4ai_project.communication.System1 import System1
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3

@dataclass
class Blackboard:
    system1: System1 = field(default_factory=System1)
    db: sqlite3.Connection | None = None   
    subscribers: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    signals: dict[int, asyncio.Event] = field(default_factory=dict)    
        
    def __post_init__(self):
        self.db = sqlite3.connect(":memory:", autocommit= True)
        self.db.executescript("""
                         CREATE TABLE IF NOT EXISTS messages (
                             id             INTEGER PRIMARY KEY AUTOINCREMENT,
                             author         text NOT NULL,
                             body           text NOT NULL,
                             created_at TEXT NOT NULL
                         )
                         """)
               
    def post_message(self, author:str, message: str):        
        created_at_iso = datetime.now(timezone.utc).isoformat()
        formatted_message = f"{author}: {message}"
        
        cursor = self.db.execute(
            "INSERT INTO messages(author, body, created_at) VALUES (?,?,?)",
            (author, formatted_message, created_at_iso)
        )
             
        topic = self.system1.infer_topic(formatted_message)
        
        for agent_idx in self.subscribers.get(topic,()):
            self.signals[agent_idx].set()        
        
        return cursor.lastrowid
    
    def read_messages(self, last_row_id_read: int | None = None):
        if last_row_id_read is None:
            cursor = self.db.execute("SELECT * FROM messages ORDER BY id ASC")
        else:
            cursor = self.db.execute(
                "SELECT * FROM messages WHERE id > ? ORDER BY id ASC",
                (last_row_id_read,),
            )
        return cursor.fetchall()
    
    def subscribe(self, agent: BlackboardAgenticPlayer):     
        self.subscribers.setdefault(agent.agent_nickname, set()).add(agent.agent_idx)   
        self.subscribers.setdefault("broadcast", set()).add(agent.agent_idx)   

        # self.system1.criteria[agent.agent_nickname] = f"Agent '{agent.agent_nickname}, with agent_idx '{agent.agent_idx}' and uuid '{agent.uuid}'"
        self.system1.criteria[agent.agent_nickname] = f"Messages addressed to {agent.agent_nickname}, agent index {agent.agent_idx}, UUID {agent.uuid}"  
        
        if agent.agent_idx not in self.signals:
            self.signals[agent.agent_idx] = asyncio.Event()    
        
    async def wait(self, agent: BlackboardAgenticPlayer, timeout_seconds: float) -> bool:
        try:
            await asyncio.wait_for(
                self.signals[agent.agent_idx].wait(),
                timeout=timeout_seconds
            )
            
            return True
        except TimeoutError:
            return False