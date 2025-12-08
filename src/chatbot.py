"""
Shared chatbot logic for agent routing and state management.
Used by both CLI and web interfaces.
"""

from typing import Dict, List, Optional, Tuple

from agents.orchestrator import Orchestrator


class ChatbotSession:
    """Manages chatbot session state using Orchestrator pattern."""

    def __init__(self):
        """Initialize orchestrator."""
        self.orchestrator = Orchestrator()

    def reset(self):
        """Reset to initial state."""
        self.orchestrator._chat_history = []
        self.orchestrator._in_order_mode = False

    def process_message(
        self, message: str, chat_history: List[Dict[str, str]]
    ) -> Tuple[str, bool, Optional[str]]:
        """Process user message and return response with metadata.

        Args:
            message: User's message
            chat_history: List of message dicts with 'role' and 'content'

        Returns:
            Tuple of (response_message, reset_history, transfer_note)
            - response_message: The agent's response text
            - reset_history: Whether to reset chat history (order completed/failed)
            - transfer_note: Optional transfer message (always None with orchestrator)
        """
        response = self.orchestrator.invoke(message, chat_history=chat_history)

        # Check if order was completed (reset history)
        reset_history = (
            "Order placed successfully" in response.message
            or "Order ID:" in response.message
        )

        return response.message, reset_history, None
