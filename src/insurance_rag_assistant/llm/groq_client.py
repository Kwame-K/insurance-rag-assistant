import json
from typing import Any

from groq import Groq

from insurance_rag_assistant.config import settings


class GroqStructuredLLMClient:
    """Groq client using strict JSON Schema structured outputs."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        resolved_api_key = api_key or settings.groq_api_key
        resolved_model_name = model_name or settings.groq_model_name

        if not resolved_api_key:
            message = (
                "GROQ_API_KEY is not configured. "
                "Add it to the project .env file or export it in your terminal."
            )
            raise RuntimeError(message)

        self.model_name = resolved_model_name
        self.client = Groq(api_key=resolved_api_key)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate JSON validated later by Pydantic and citation checks."""
        required_fields = ", ".join(response_schema.get("required", []))

        response_contract = f"""
Return exactly one valid JSON object and no Markdown.

The required top-level keys are:
{required_fields}

Use this structure:

{{
  "answer": "string",
  "answer_language": "en or fr",
  "grounded": true,
  "confidence": "high, medium, or low",
  "citations": [
    {{
      "chunk_id": "string",
      "document_id": "string",
      "document_name": "string",
      "section_title": "string",
      "page_start": null,
      "page_end": null,
      "quote": "short verbatim source excerpt",
      "retrieval_score": 0.0
    }}
  ],
  "insufficient_context_reason": null
}}

Rules for the JSON object:
- Include every top-level key exactly once.
- If grounded is true, citations must contain at least one citation.
- If grounded is false, citations must be an empty list.
- If grounded is false, insufficient_context_reason must be a non-empty string.
- If grounded is true, insufficient_context_reason must be null.
- Use null for page_start and page_end when the source is Markdown.
- Keep each quote short, verbatim, and copied from the cited passage.
""".strip()

        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_completion_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": (f"{system_prompt}\n\n{response_contract}"),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_object",
            },
        )

        content = completion.choices[0].message.content

        if content is None:
            raise RuntimeError("Groq returned an empty response.")

        parsed_content = json.loads(content)

        if not isinstance(parsed_content, dict):
            raise TypeError("Groq response must be a JSON object.")

        return parsed_content
