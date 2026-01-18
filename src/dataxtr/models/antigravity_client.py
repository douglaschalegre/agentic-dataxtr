"""Antigravity API client for LangChain.

This module provides a LangChain-compatible chat model that uses Google's
Antigravity API to access Claude and Gemini models via OAuth authentication.

Prerequisites:
    - Install and authenticate with opencode-antigravity-auth plugin
    - Run `opencode auth login` to create tokens at ~/.config/opencode/antigravity-accounts.json
"""

import asyncio
import json
import os
import random
import time
import uuid

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import Any, Optional

import aiohttp
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

# ============================================================================
# Constants from opencode-antigravity-auth
#
# NOTE: client id/secret are treated as configuration and loaded from the
# environment to avoid hard-coding sensitive values in the repository.
# ============================================================================

ANTIGRAVITY_CLIENT_ID = os.getenv("ANTIGRAVITY_CLIENT_ID", "")
ANTIGRAVITY_CLIENT_SECRET = os.getenv("ANTIGRAVITY_CLIENT_SECRET", "")
ANTIGRAVITY_ENDPOINT = os.getenv("ANTIGRAVITY_ENDPOINT", "")
ANTIGRAVITY_DEFAULT_PROJECT_ID = os.getenv("ANTIGRAVITY_DEFAULT_PROJECT_ID", "")

ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.11.5 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": (
        '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}'
    ),
}

# Token expiry buffer (1 minute before actual expiry)
ACCESS_TOKEN_EXPIRY_BUFFER_MS = 60 * 1000

# Default accounts file location
DEFAULT_ACCOUNTS_PATH = Path.home() / ".config" / "opencode" / "antigravity-accounts.json"


class AntigravityAuthError(Exception):
    """Raised when authentication fails."""

    pass


class AntigravityAPIError(Exception):
    """Raised when API call fails."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ChatAntigravity(BaseChatModel):
    retry_429_timeout_s: float = Field(
        default=60.0,
        description="Max time to keep retrying on HTTP 429 in seconds",
    )
    """LangChain chat model using Google's Antigravity API.

    This model reads OAuth credentials from ~/.config/opencode/antigravity-accounts.json
    (created by opencode-antigravity-auth plugin) and uses them to access Claude and
    Gemini models through Google's Antigravity gateway.

    Example:
        >>> from dataxtr.models.antigravity_client import ChatAntigravity
        >>> model = ChatAntigravity(model="claude-sonnet-4-5")
        >>> response = model.invoke([HumanMessage(content="Hello!")])
        >>> print(response.content)

    Attributes:
        model: The model ID (e.g., "claude-sonnet-4-5", "gemini-3-pro-high")
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum output tokens
        thinking_budget: Token budget for thinking models (e.g., claude-sonnet-4-5-thinking)
        accounts_path: Path to antigravity-accounts.json (default: ~/.config/opencode/)
    """

    model: str = Field(default="claude-sonnet-4-5", description="Antigravity model ID")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=8192, description="Maximum output tokens")
    thinking_budget: Optional[int] = Field(
        default=None, description="Thinking budget for thinking models"
    )
    accounts_path: Path = Field(
        default=DEFAULT_ACCOUNTS_PATH, description="Path to antigravity-accounts.json"
    )

    # Private state for token caching
    _access_token: Optional[str] = PrivateAttr(default=None)
    _token_expires: int = PrivateAttr(default=0)  # Unix timestamp in milliseconds
    _project_id: Optional[str] = PrivateAttr(default=None)
    _refresh_token: Optional[str] = PrivateAttr(default=None)
    _credentials_loaded: bool = PrivateAttr(default=False)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "antigravity"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for caching."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking_budget": self.thinking_budget,
        }

    def _load_credentials(self) -> None:
        """Load credentials from antigravity-accounts.json.

        Reads the first/active account from the accounts file created by
        opencode-antigravity-auth plugin.

        Raises:
            AntigravityAuthError: If credentials file is missing or invalid
        """
        if self._credentials_loaded:
            return

        if not self.accounts_path.exists():
            raise AntigravityAuthError(
                f"Antigravity accounts file not found at {self.accounts_path}\n"
                "Please install opencode-antigravity-auth and run 'opencode auth login'"
            )

        try:
            data = json.loads(self.accounts_path.read_text())
        except json.JSONDecodeError as e:
            raise AntigravityAuthError(f"Invalid JSON in accounts file: {e}")

        accounts = data.get("accounts", [])
        if not accounts:
            raise AntigravityAuthError(
                "No accounts found in antigravity-accounts.json\n"
                "Please run 'opencode auth login' to authenticate"
            )

        # Use active account or first account
        active_index = data.get("activeIndex", 0)
        if active_index >= len(accounts):
            active_index = 0

        account = accounts[active_index]
        self._refresh_token = account.get("refreshToken")
        if not self._refresh_token:
            raise AntigravityAuthError("No refresh token found in account")

        # Get project ID (prefer managedProjectId, then projectId, then default)
        self._project_id = (
            account.get("managedProjectId")
            or account.get("projectId")
            or ANTIGRAVITY_DEFAULT_PROJECT_ID
        )

        self._credentials_loaded = True

    def _access_token_expired(self) -> bool:
        """Check if access token is expired or missing.

        Uses a 1-minute buffer before actual expiry to handle clock skew
        and network latency.

        Returns:
            True if token is expired or missing, False otherwise
        """
        if not self._access_token or self._token_expires == 0:
            return True
        return self._token_expires <= int(time.time() * 1000) + ACCESS_TOKEN_EXPIRY_BUFFER_MS

    async def _refresh_access_token(self) -> str:
        """Refresh the OAuth access token.

        Makes a request to Google's OAuth endpoint to exchange the refresh token
        for a new access token. Updates internal state with the new token.

        Returns:
            The new access token

        Raises:
            AntigravityAuthError: If token refresh fails
        """
        self._load_credentials()

        start_time = int(time.time() * 1000)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()

                    # Check for revoked token
                    if "invalid_grant" in error_text:
                        raise AntigravityAuthError(
                            "OAuth token has been revoked by Google.\n"
                            "Please run 'opencode auth login' to re-authenticate."
                        )

                    raise AntigravityAuthError(
                        f"Token refresh failed ({resp.status}): {error_text}"
                    )

                data = await resp.json()
                self._access_token = data["access_token"]
                # Calculate expiry: start_time + expires_in * 1000
                expires_in = data.get("expires_in", 3600)
                self._token_expires = start_time + (expires_in * 1000)

                return self._access_token  # type: ignore[return-value]

    async def _ensure_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            A valid access token

        Raises:
            AntigravityAuthError: If unable to get a valid token
        """
        if not self._access_token_expired():
            return self._access_token  # type: ignore

        return await self._refresh_access_token()  # type: ignore[return-value]

    def _convert_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """Convert LangChain messages to Antigravity format.

        Antigravity uses Gemini-style format:
        - Role is "user" or "model" (not "assistant")
        - Content is in "parts" array with "text" objects
        - System instructions are separate from contents

        Args:
            messages: List of LangChain messages

        Returns:
            Tuple of (contents array, system_instruction or None)
        """
        contents: list[dict[str, Any]] = []
        system_instruction: Optional[dict[str, Any]] = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System instruction must be object with parts array
                system_instruction = {"parts": [{"text": str(msg.content)}]}

            elif isinstance(msg, HumanMessage):
                contents.append({"role": "user", "parts": [{"text": str(msg.content)}]})

            elif isinstance(msg, AIMessage):
                # Role is "model" not "assistant" in Gemini format
                parts: list[dict[str, Any]] = []

                if msg.content:
                    parts.append({"text": str(msg.content)})

                # Handle tool calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "id": tc.get("id", ""),
                                }
                            }
                        )

                if parts:
                    contents.append({"role": "model", "parts": parts})

            elif isinstance(msg, ToolMessage):
                # Tool responses go as user messages with functionResponse
                response_content = msg.content
                if isinstance(response_content, str):
                    try:
                        response_content = json.loads(response_content)
                    except json.JSONDecodeError:
                        response_content = {"result": response_content}

                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "",
                                    "id": msg.tool_call_id or "",
                                    "response": response_content,
                                }
                            }
                        ],
                    }
                )

        return contents, system_instruction

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert LangChain tools to Antigravity functionDeclarations format.

        Args:
            tools: List of tool definitions in LangChain format

        Returns:
            List with single object containing functionDeclarations
        """
        function_declarations = []

        for tool in tools:
            declaration = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }

            # Add parameters if present
            if "parameters" in tool:
                declaration["parameters"] = tool["parameters"]

            function_declarations.append(declaration)

        return [{"functionDeclarations": function_declarations}]

    def _parse_response(self, response_data: dict[str, Any]) -> ChatResult:
        """Parse Antigravity API response into LangChain ChatResult.

        Args:
            response_data: Raw API response

        Returns:
            ChatResult with generations

        Raises:
            AntigravityAPIError: If response format is unexpected
        """
        response = response_data.get("response", {})
        candidates = response.get("candidates", [])

        if not candidates:
            raise AntigravityAPIError(
                "No candidates in API response", response_body=str(response_data)
            )

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text and tool calls from parts
        text_content = ""
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    {
                        "name": fc.get("name", ""),
                        "args": fc.get("args", {}),
                        "id": fc.get("id", str(uuid.uuid4())),
                    }
                )

        # Build AIMessage
        message = AIMessage(
            content=text_content,
            tool_calls=tool_calls,
        )

        # Extract usage metadata
        usage_metadata = response.get("usageMetadata", {})
        generation_info = {
            "finish_reason": candidate.get("finishReason", "STOP"),
            "model_version": response.get("modelVersion", self.model),
            "response_id": response.get("responseId", ""),
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

        return ChatResult(
            generations=[ChatGeneration(message=message, generation_info=generation_info)],
            llm_output=generation_info,
        )

    async def _make_api_request(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Make a request to the Antigravity API.

        Args:
            messages: List of messages to send
            tools: Optional list of tool definitions

        Returns:
            Raw API response

        Raises:
            AntigravityAPIError: If API call fails
        """
        self._load_credentials()
        contents, system_instruction = self._convert_messages(messages)

        # Build request body
        request_body: dict[str, Any] = {
            "project": self._project_id,
            "model": self.model,
            "request": {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            },
            "userAgent": "antigravity",
            "requestId": f"dataxtr-{uuid.uuid4().hex[:12]}",
        }

        if system_instruction:
            request_body["request"]["systemInstruction"] = system_instruction

        # Add thinking config for thinking models
        if self.thinking_budget and "thinking" in self.model:
            request_body["request"]["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget,
                "includeThoughts": False,
            }

        if tools:
            converted_tools = self._convert_tools(tools)
            request_body["request"]["tools"] = converted_tools
            # Debug: help diagnose schema incompatibilities
            try:
                first = converted_tools[0]["functionDeclarations"][0].get("parameters", {})
                if isinstance(first, dict) and ("$defs" in first or "$ref" in first):
                    print(
                        "[antigravity] warning: unsanitized schema keys in tool parameters: "
                        f"{[k for k in ['$defs', '$ref'] if k in first]}"
                    )
            except Exception:
                pass

        # Get access token and make request
        access_token = await self._ensure_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_HEADERS,
        }

        timeout_s = max(0.0, float(self.retry_429_timeout_s))
        start_ms = int(time.time() * 1000)

        async with aiohttp.ClientSession() as session:
            attempt = 0
            while True:
                attempt += 1
                async with session.post(
                    f"{ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                    json=request_body,
                    headers=headers,
                ) as resp:
                    response_text = await resp.text()

                    if resp.ok:
                        return json.loads(response_text)

                    # Parse error response
                    error_msg = f"Antigravity API error ({resp.status})"
                    try:
                        error_data = json.loads(response_text)
                        if "error" in error_data:
                            error = error_data["error"]
                            error_msg = f"{error_msg}: {error.get('message', response_text)}"
                    except json.JSONDecodeError:
                        error_msg = f"{error_msg}: {response_text}"

                    if resp.status == 429 and timeout_s > 0:
                        elapsed_s = (int(time.time() * 1000) - start_ms) / 1000.0
                        remaining_s = timeout_s - elapsed_s
                        if remaining_s <= 0:
                            raise AntigravityAPIError(
                                error_msg, status_code=resp.status, response_body=response_text
                            )

                        retry_after_hdr = resp.headers.get("Retry-After")
                        retry_after_s: Optional[float] = None
                        if retry_after_hdr:
                            try:
                                retry_after_s = float(retry_after_hdr)
                            except ValueError:
                                retry_after_s = None

                        # Exponential backoff with jitter, capped.
                        backoff_s = min(10.0, 0.5 * (2 ** (attempt - 1)))
                        jitter_s = random.uniform(0.0, 0.25)
                        sleep_s = (
                            retry_after_s if retry_after_s is not None else (backoff_s + jitter_s)
                        )
                        sleep_s = min(sleep_s, max(0.0, remaining_s))

                        if sleep_s > 0:
                            await asyncio.sleep(sleep_s)
                            continue

                    raise AntigravityAPIError(
                        error_msg, status_code=resp.status, response_body=response_text
                    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response synchronously.

        This is a wrapper around the async implementation using asyncio.

        Args:
            messages: List of messages
            stop: Stop sequences (currently not supported)
            run_manager: Callback manager
            **kwargs: Additional arguments

        Returns:
            ChatResult with generated response
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, need to use nest_asyncio or similar
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                self._agenerate(messages, stop, run_manager, **kwargs)
            )
        else:
            return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response asynchronously.

        Args:
            messages: List of messages
            stop: Stop sequences (currently not supported)
            run_manager: Callback manager
            **kwargs: Additional arguments (may include 'tools')

        Returns:
            ChatResult with generated response
        """
        tools = kwargs.get("tools")
        response_data = await self._make_api_request(messages, tools)
        return self._parse_response(response_data)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ChatAntigravity":
        """Bind tools to this model.

        Args:
            tools: List of tools (BaseTool instances or dicts)
            **kwargs: Additional arguments

        Returns:
            A new ChatAntigravity instance with tools bound
        """
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel

        def _sanitize_json_schema(obj: Any) -> Any:
            # Antigravity (Gemini-style) functionDeclarations schema doesn't accept $defs/$ref.
            # We also drop `title` to reduce noise.
            if isinstance(obj, dict):
                # If this is a ref-only node, degrade it to an unconstrained object.
                if "$ref" in obj and len(obj) == 1:
                    return {"type": "object"}

                cleaned: dict[str, Any] = {}
                for k, v in obj.items():
                    if k in {"$defs", "$ref", "title"}:
                        continue
                    # Some schemas embed refs inside `anyOf` / `items`; sanitize recursively.
                    cleaned[k] = _sanitize_json_schema(v)
                return cleaned
            if isinstance(obj, list):
                return [_sanitize_json_schema(v) for v in obj]
            return obj

        def _pydantic_parameters(schema: Any) -> dict[str, Any]:
            if schema is None:
                return {}
            raw: Any
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                raw = schema.model_json_schema()
            elif hasattr(schema, "model_json_schema"):
                raw = schema.model_json_schema()  # type: ignore[attr-defined]
            else:
                raw = {}
            return _sanitize_json_schema(raw)

        # Convert tools to dict format
        tool_dicts: list[dict[str, Any]] = []
        for tool in tools:
            # Standard LC tool instances
            if isinstance(tool, BaseTool):
                tool_dicts.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _pydantic_parameters(tool.args_schema),
                    }
                )
                continue

            # Some toolkits pass BaseTool classes instead of instances.
            if isinstance(tool, type) and issubclass(tool, BaseTool):
                tool_dicts.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _pydantic_parameters(getattr(tool, "args_schema", None)),
                    }
                )
                continue

            # LangChain structured output sometimes binds a Pydantic model/class as a tool.
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                tool_dicts.append(
                    {
                        "name": tool.__name__,
                        "description": tool.__doc__ or "Structured output schema",
                        "parameters": _pydantic_parameters(tool),
                    }
                )
                continue

            # Raw dict tool definition
            if isinstance(tool, dict):
                tool_dicts.append(tool)
                continue

            # Last resort: tool-like object
            if hasattr(tool, "name") and hasattr(tool, "description"):
                args_schema = getattr(tool, "args_schema", None)
                tool_dicts.append(
                    {
                        "name": str(getattr(tool, "name")),
                        "description": str(getattr(tool, "description")),
                        "parameters": _pydantic_parameters(args_schema),
                    }
                )
                continue

            raise ValueError(f"Unsupported tool type: {type(tool)}")

        # Create a subclass that includes tools in every request
        parent = self

        class BoundChatAntigravity(ChatAntigravity):
            _bound_tools: list[dict] = PrivateAttr(default_factory=list)

            def __init__(self, **data: Any):
                super().__init__(**data)
                self._bound_tools = tool_dicts
                # Copy parent state
                self._access_token = parent._access_token
                self._token_expires = parent._token_expires
                self._project_id = parent._project_id
                self._refresh_token = parent._refresh_token
                self._credentials_loaded = parent._credentials_loaded

            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
                tools = kwargs.get("tools", self._bound_tools)
                response_data = await self._make_api_request(messages, tools)
                return self._parse_response(response_data)

        return BoundChatAntigravity(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_budget=self.thinking_budget,
            accounts_path=self.accounts_path,
        )
