from dataclasses import dataclass, field
import os
from typesafe_sdk import Choice, TypeSafeClient

@dataclass
class System1:  
    criteria: dict[str,str] = field(default_factory=dict)
    client: TypeSafeClient | None = None
    threshold: float = 70
    
    def __post_init__(self):   
        self.client = TypeSafeClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api",
        )
        
        self.criteria["broadcast"] = "broadcast channel:  when there is no specific destination, messages should be routed/sent to broadcast"
           
    def infer_topic(self, message: str):        
        result = self.client.system_one(
            state=message,
            model="jev-latest",
            questions={
                "destination": Choice(
                    instructions="Which agent should handle this message? Send to broadcast if you are unsure or there is no destination in the message",
                    criteria=self.criteria
                )
            }
        )
        
        answer = result.answers["destination"]
        
        if(answer.confidence < self.threshold or answer.choice not in self.criteria):           
            return "broadcast"
  
        return answer.choice