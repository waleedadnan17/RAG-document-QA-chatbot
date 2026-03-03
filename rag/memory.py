"""Conversation memory utilities."""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConversationMessage:
    """Represents a message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class ConversationMemory:
    """Simple conversation memory with rolling window."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep
        """
        self.max_history = max_history
        self.messages: List[ConversationMessage] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append(ConversationMessage(role=role, content=content))
        
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self) -> str:
        """Get conversation context for LLM."""
        if not self.messages:
            return ""
        
        context_lines = []
        for msg in self.messages:
            if msg.role == "user":
                context_lines.append(f"User: {msg.content}")
            else:
                context_lines.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_lines)
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_messages(self) -> List[Tuple[str, str]]:
        """Get all messages as (role, content) tuples."""
        return [(msg.role, msg.content) for msg in self.messages]
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
