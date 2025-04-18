import json
import os
from datetime import datetime

class ChatHistory:
    def __init__(self):
        self.history_dir = "chat_history"
        os.makedirs(self.history_dir, exist_ok=True)
        
        self.current_session = []
        self.max_history = 10
        
    def add_message(self, role, content):
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.current_session.append(message)
        self.save_session()
        
    def get_recent_messages(self):
        
        messages = []
        for msg in self.current_session[-self.max_history:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages
        
    def save_session(self):
        
        if not self.current_session:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, ensure_ascii=False, indent=2)
