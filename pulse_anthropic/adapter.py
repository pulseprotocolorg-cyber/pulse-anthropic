"""Anthropic Claude adapter for PULSE Protocol.

Translates PULSE semantic messages to Anthropic Messages API and back.
Same interface as OpenAIAdapter — swap provider in one line of code.

Example:
    >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
    >>> msg = PulseMessage(
    ...     action="ACT.QUERY.DATA",
    ...     parameters={"query": "Explain quantum computing"}
    ... )
    >>> response = adapter.send(msg)
    >>> print(response.content["parameters"]["result"])
"""

from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError, AuthenticationError

from pulse.message import PulseMessage
from pulse.adapter import PulseAdapter, AdapterError, AdapterConnectionError


# Map PULSE actions to Claude system prompts
ACTION_PROMPTS = {
    "ACT.ANALYZE.SENTIMENT": (
        "Analyze the sentiment of the following text. "
        "Respond with a JSON object: "
        '{"sentiment": "positive|negative|neutral|mixed", '
        '"confidence": 0.0-1.0, "explanation": "brief reason"}'
    ),
    "ACT.ANALYZE.PATTERN": (
        "Analyze the following data for patterns and trends. "
        "Respond with a JSON object: "
        '{"patterns": ["pattern1", ...], "summary": "brief summary"}'
    ),
    "ACT.CREATE.TEXT": (
        "Generate text based on the following instructions."
    ),
    "ACT.TRANSFORM.TRANSLATE": (
        "Translate the following text. "
        "Output only the translation, nothing else."
    ),
    "ACT.TRANSFORM.SUMMARIZE": (
        "Summarize the following text concisely."
    ),
}

# Default model for each action type
DEFAULT_MODELS = {
    "ACT.ANALYZE.SENTIMENT": "claude-haiku-4-5-20251001",
    "ACT.ANALYZE.PATTERN": "claude-sonnet-4-5-20250929",
    "ACT.CREATE.TEXT": "claude-sonnet-4-5-20250929",
    "ACT.QUERY.DATA": "claude-haiku-4-5-20251001",
    "ACT.TRANSFORM.TRANSLATE": "claude-haiku-4-5-20251001",
    "ACT.TRANSFORM.SUMMARIZE": "claude-haiku-4-5-20251001",
}


class AnthropicAdapter(PulseAdapter):
    """PULSE adapter for Anthropic Claude API.

    Translates PULSE semantic actions to Anthropic Messages API calls.
    Same interface as OpenAIAdapter — switching providers is one line change.

    Supported PULSE actions:
        - ACT.QUERY.DATA — ask a question, get an answer
        - ACT.CREATE.TEXT — generate text from instructions
        - ACT.ANALYZE.SENTIMENT — analyze text sentiment
        - ACT.ANALYZE.PATTERN — find patterns in data
        - ACT.TRANSFORM.TRANSLATE — translate text
        - ACT.TRANSFORM.SUMMARIZE — summarize text

    Example:
        >>> # Same code as OpenAI — just change the adapter
        >>> # adapter = OpenAIAdapter(api_key="sk-...")
        >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
        >>> msg = PulseMessage(
        ...     action="ACT.ANALYZE.SENTIMENT",
        ...     parameters={"text": "I love this product!"}
        ... )
        >>> response = adapter.send(msg)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            name="anthropic",
            base_url=base_url or "https://api.anthropic.com",
            config=config or {},
        )
        self.default_model = model
        self._client: Optional[Anthropic] = None
        self._api_key = api_key

    def connect(self) -> None:
        """Initialize Anthropic client.

        Raises:
            AdapterConnectionError: If API key is missing
        """
        if not self._api_key:
            raise AdapterConnectionError(
                "Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY env var."
            )

        try:
            self._client = Anthropic(
                api_key=self._api_key,
                base_url=self.base_url if self.base_url != "https://api.anthropic.com" else None,
            )
            self.connected = True
        except Exception as e:
            raise AdapterConnectionError(f"Failed to initialize Anthropic client: {e}") from e

    def disconnect(self) -> None:
        """Close Anthropic client."""
        self._client = None
        self.connected = False

    def to_native(self, message: PulseMessage) -> Dict[str, Any]:
        """Convert PULSE message to Anthropic Messages API request.

        Args:
            message: PULSE message with action and parameters

        Returns:
            Dictionary ready for Anthropic messages.create()
        """
        action = message.content["action"]
        params = message.content.get("parameters", {})

        # Build user message
        user_content = self._build_user_content(action, params)

        # Build system prompt
        system_prompt = self._build_system_prompt(action, params)

        # Select model
        model = params.get("model", DEFAULT_MODELS.get(action, self.default_model))

        # Build request
        request = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": params.get("max_tokens", 1024),
        }

        if system_prompt:
            request["system"] = system_prompt

        if "temperature" in params:
            request["temperature"] = params["temperature"]

        return request

    def call_api(self, native_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Anthropic Messages API.

        Args:
            native_request: Anthropic-formatted request dict

        Returns:
            Dictionary with response data
        """
        if not self._client:
            self.connect()

        try:
            response = self._client.messages.create(**native_request)

            # Extract text from content blocks
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            return {
                "content": text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

        except AuthenticationError as e:
            raise AdapterError(
                f"Anthropic authentication failed. Check your API key. {e}"
            ) from e
        except RateLimitError as e:
            raise AdapterError(
                f"Anthropic rate limit exceeded. Retry later. {e}"
            ) from e
        except APIConnectionError as e:
            raise AdapterConnectionError(
                f"Cannot reach Anthropic API: {e}"
            ) from e
        except APIError as e:
            error_code = self.map_error_code(e.status_code) if e.status_code else "META.ERROR.UNKNOWN"
            raise AdapterError(
                f"Anthropic API error ({error_code}): {e}"
            ) from e

    def from_native(self, native_response: Dict[str, Any]) -> PulseMessage:
        """Convert Anthropic response to PULSE message.

        Args:
            native_response: Response dict from call_api()

        Returns:
            PULSE response message with result and metadata
        """
        return PulseMessage(
            action="ACT.RESPOND",
            parameters={
                "result": native_response["content"],
                "model": native_response["model"],
                "usage": native_response["usage"],
                "stop_reason": native_response["stop_reason"],
            },
            validate=False,
        )

    @property
    def supported_actions(self) -> List[str]:
        """Actions this adapter supports."""
        return [
            "ACT.QUERY.DATA",
            "ACT.CREATE.TEXT",
            "ACT.ANALYZE.SENTIMENT",
            "ACT.ANALYZE.PATTERN",
            "ACT.TRANSFORM.TRANSLATE",
            "ACT.TRANSFORM.SUMMARIZE",
        ]

    def _build_user_content(self, action: str, params: Dict[str, Any]) -> str:
        """Build user message content from PULSE parameters."""
        if "query" in params:
            return params["query"]

        if "text" in params:
            text = params["text"]
            if action == "ACT.TRANSFORM.TRANSLATE":
                target_lang = params.get("target_language", "English")
                return f"Translate to {target_lang}:\n\n{text}"
            return text

        if "instructions" in params:
            return params["instructions"]

        if "data" in params:
            return f"Analyze this data:\n\n{params['data']}"

        if "prompt" in params:
            return params["prompt"]

        raise AdapterError(
            f"Missing content parameter. For {action}, provide one of: "
            "'query', 'text', 'instructions', 'data', or 'prompt'."
        )

    def _build_system_prompt(self, action: str, params: Dict[str, Any]) -> Optional[str]:
        """Build system prompt based on PULSE action."""
        if "system_prompt" in params:
            return params["system_prompt"]

        if action in ACTION_PROMPTS:
            return ACTION_PROMPTS[action]

        return None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AnthropicAdapter(model='{self.default_model}', "
            f"connected={self.connected})"
        )
