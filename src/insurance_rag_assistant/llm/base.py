from typing import Any, Protocol


class StructuredLLMClient(Protocol):
    """Interface for an LLM that returns JSON matching a supplied schema."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a structured JSON response."""
