from contextlib import contextmanager
from queue import Queue
from typing import Any, Iterator
from a4ai_project.communication.RouterSystem1 import RouterSystem1
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


def default_blackboard_path() -> Path:
    return Path("blackboard-v2.sqlite")


@dataclass
class BoardMessage:
    id:int
    author: str
    body: str
    created_at: str
    
    def get_formatted_messsage(self):
        return f"{self.author}: {self.body}"

@dataclass
class Blackboard:
    router: RouterSystem1 = field(default_factory=RouterSystem1)
    inboxes_by_topic: dict[str, Queue] = field(default_factory=dict)
    db_path: Path = field(default_factory=default_blackboard_path)
        
    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute("DELETE FROM messages")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection on the calling thread and close it before returning."""
        connection = sqlite3.connect(self.db_path, autocommit=True)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
        finally:
            connection.close()
    
    def subscribe(self, agent: Any) -> Queue: 
        queue =  Queue()    
        self.inboxes_by_topic[agent.agent_nickname] =  queue   
        self.router.criteria[agent.agent_nickname] = f"Messages addressed to {agent.agent_nickname}, agent index {agent.agent_idx}, UUID {agent.uuid}"  
            
        return queue
                           
    def _publish_message(self, author:str, message: str, destination: str):        
        created_at_iso = datetime.now(timezone.utc).isoformat()
        
        with self._connection() as db:
            cursor = db.execute(
                "INSERT INTO messages(author, body, created_at) VALUES (?,?,?)",
                (author, message, created_at_iso)
            )
            message_id = cursor.lastrowid

        board_message = BoardMessage(message_id, author, message, created_at=created_at_iso)      
        
        if destination == "broadcast":
            for topic, inbox in self.inboxes_by_topic.items():
                if topic != author:
                    inbox.put(board_message)                    
        elif destination in self.inboxes_by_topic:
            self.inboxes_by_topic[destination].put(board_message)
        else: 
            raise ValueError(f"Wrong destination inbox: {destination}")
                         
        return board_message
    
    def post_message(self, author:str, message: str) -> BoardMessage:
        destination = self.router.infer_destination(f"{author}: {message}")
        message = self._publish_message(author=author, message = message, destination = destination)
        return message        
        
    def broadcast_message(self, author:str, message: str) -> BoardMessage:
        message = self._publish_message(author=author, message = message, destination = "broadcast")
        return message
    
    def read_messages(self, limit: int | None = None):
        with self._connection() as db:
            if limit is None:
                cursor = db.execute("SELECT * FROM messages ORDER BY id DESC")
            else:
                cursor = db.execute(
                    "SELECT * FROM messages ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return cursor.fetchall()
    
    # async def wait(self, agent: Any, timeout_seconds: float) -> bool:
    #     try:
    #         await asyncio.wait_for(
    #             self.inboxes[agent.agent_idx].wait(),
    #             timeout=timeout_seconds
    #         )
    #         
    #         return True
    #     except TimeoutError:
    #         return False
