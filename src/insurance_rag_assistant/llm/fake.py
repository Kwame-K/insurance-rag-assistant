from typing import Any


class FakeStructuredLLMClient:
    """Deterministic LLM replacement used in unit tests."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.response_schema: dict[str, Any] | None = None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_schema = response_schema

        return self.response
