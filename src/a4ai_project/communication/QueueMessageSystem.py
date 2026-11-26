
# from dataclasses import dataclass
# import queue

# @dataclass
# class QueueMessageSystem():
     
#     def subscribe(self, topic: str) -> queue.Queue:   
#         self.topics.setdefault(topic, set()).add(agent.agent_idx)   
#         self.topics.setdefault("broadcast", set()).add(agent.agent_idx)   
#         self.inboxes[topic] = queue.Queue
            
#         return self.inboxes[agent.agent_idx]