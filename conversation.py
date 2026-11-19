"""Conversation between a user and a language model."""

from __future__ import annotations

from models import LanguageModel


class Conversation:
    """Stores the messages exchanged with one language model."""

    def __init__(self, conversation_id: int, model: LanguageModel) -> None:
        """Create an empty conversation that uses the given model."""
        self.conversation_id = conversation_id
        self.model = model
        self.messages: list[dict[str, object]] = []

    def send_message(self, prompt: str) -> str | list[float]:
        """Send a prompt to the model and store both the prompt and the response."""
        response = self.model.generate_response(prompt)
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})
        return response

    def display_history(self) -> None:
        """Print every message in the conversation."""
        print(f"--- Conversation #{self.conversation_id} with {self.model} ---")
        if not self.messages:
            print("(no messages yet)")
        for message in self.messages:
            print(f"[{message['role']}] {message['content']}")

    def __len__(self) -> int:
        """Return the number of stored messages."""
        return len(self.messages)
