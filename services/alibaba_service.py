import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx

from config.settings import settings
from prompts.system_prompt import SYSTEM_PROMPT
from utils.logger import logger


FREE_QUOTA_CODE = "AllocationQuota.FreeTierOnly"


# ─────────────────────────────────────
# Exceptions
# ─────────────────────────────────────

class AlibabaServiceError(Exception):
    pass


class AlibabaNotConfiguredError(AlibabaServiceError):
    pass


class AlibabaProviderUnavailableError(AlibabaServiceError):
    pass


class AlibabaApplicationError(AlibabaServiceError):
    pass


class AlibabaPartialStreamError(AlibabaServiceError):
    pass


class AllAlibabaModelsFailed(AlibabaServiceError):
    pass


# ─────────────────────────────────────
# Model state
# ─────────────────────────────────────

@dataclass
class ModelState:
    status: str = "available"
    cooldown_until: float = 0.0
    reason: str = ""


# ─────────────────────────────────────
# Alibaba Service
# ─────────────────────────────────────

class AlibabaService:
    def __init__(self):
        self.api_key = (
            settings.alibaba_api_key.strip()
        )

        self.base_url = (
            settings.alibaba_base_url
            .rstrip("/")
        )

        self.models = [
            model.strip()
            for model in settings.alibaba_models.split(",")
            if model.strip()
        ]

        self.timeout_seconds = float(
            settings.llm_timeout_seconds
        )

        # Cooldowns are local only.
        # No extra settings needed.
        self.transient_cooldown = 30.0
        self.rate_limit_cooldown = 60.0
        self.access_cooldown = 300.0

        self._states: Dict[str, ModelState] = {
            model: ModelState()
            for model in self.models
        }

        self._client: Optional[
            httpx.AsyncClient
        ] = None

        if not self.api_key:
            logger.warning(
                "Alibaba API key is not configured"
            )

        if not self.models:
            logger.warning(
                "No Alibaba models configured"
            )

    # ─────────────────────────────────
    # HTTP client
    # ─────────────────────────────────

    def _get_client(
        self,
    ) -> httpx.AsyncClient:

        if self._client is None:
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=min(
                    3.0,
                    self.timeout_seconds,
                ),
            )

            self._client = (
                httpx.AsyncClient(
                    timeout=timeout
                )
            )

        return self._client

    @property
    def endpoint(self) -> str:
        return (
            f"{self.base_url}/chat/completions"
        )

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

    # ─────────────────────────────────
    # Error helpers
    # ─────────────────────────────────

    @staticmethod
    def _error_details(
        body: str,
    ) -> Tuple[str, str]:

        try:
            payload = json.loads(body)
        except Exception:
            return "", body[:200]

        error = payload.get(
            "error",
            payload,
        )

        if isinstance(error, dict):
            code = str(
                error.get("code")
                or error.get("type")
                or ""
            )

            message = str(
                error.get("message")
                or ""
            )

            return code, message

        return "", str(error)[:200]

    @staticmethod
    def _is_quota_exhausted(
        code: str,
        message: str,
    ) -> bool:

        combined = (
            f"{code} {message}"
        ).lower()

        return (
            FREE_QUOTA_CODE.lower()
            in combined
            or (
                "free quota" in combined
                and "exhaust" in combined
            )
            or "free tier only" in combined
            or "free-tier only" in combined
        )

    @staticmethod
    def _looks_model_unavailable(
        code: str,
        message: str,
    ) -> bool:

        combined = (
            f"{code} {message}"
        ).lower()

        phrases = (
            "model not found",
            "model_not_found",
            "model unavailable",
            "model is unavailable",
            "model does not exist",
            "model not available",
            "model access",
            "not activated",
        )

        return any(
            phrase in combined
            for phrase in phrases
        )

    # ─────────────────────────────────
    # Model state
    # ─────────────────────────────────

    def _mark_quota_exhausted(
        self,
        model: str,
    ) -> None:

        state = self._states[model]

        state.status = "quota_exhausted"
        state.cooldown_until = 0.0
        state.reason = FREE_QUOTA_CODE

        logger.warning(
            f"Alibaba model {model}: "
            "free quota exhausted — disabled"
        )

    def _mark_cooldown(
        self,
        model: str,
        status: str,
        seconds: float,
        reason: str,
    ) -> None:

        state = self._states[model]

        state.status = status
        state.cooldown_until = (
            time.monotonic()
            + seconds
        )
        state.reason = reason

        logger.warning(
            f"Alibaba model {model}: "
            f"{status} — cooldown "
            f"{seconds:.0f}s"
        )

    def _should_skip(
        self,
        model: str,
    ) -> Optional[str]:

        state = self._states[model]

        if state.status == "quota_exhausted":
            return "quota exhausted"

        if (
            state.cooldown_until
            > time.monotonic()
        ):
            remaining = (
                state.cooldown_until
                - time.monotonic()
            )

            return (
                f"{state.status}, "
                f"{remaining:.0f}s cooldown"
            )

        if state.status != "available":
            self._states[model] = (
                ModelState()
            )

        return None

    # ─────────────────────────────────
    # HTTP error handling
    # ─────────────────────────────────

    def _handle_http_error(
        self,
        model: str,
        status_code: int,
        body: str,
    ) -> None:

        code, message = (
            self._error_details(body)
        )

        if self._is_quota_exhausted(
            code,
            message,
        ):
            self._mark_quota_exhausted(
                model
            )
            return

        if status_code == 401:
            raise (
                AlibabaProviderUnavailableError(
                    "Alibaba authentication "
                    "failed"
                )
            )

        if status_code in {
            400,
            422,
        }:
            if self._looks_model_unavailable(
                code,
                message,
            ):
                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.access_cooldown
                    ),
                    reason=(
                        f"{status_code}:{code}"
                    ),
                )
                return

            raise AlibabaApplicationError(
                f"Alibaba rejected request "
                f"({status_code}, "
                f"{code or 'invalid_request'})"
            )

        if status_code in {
            403,
            404,
        }:
            self._mark_cooldown(
                model=model,
                status=(
                    "temporarily_unavailable"
                ),
                seconds=(
                    self.access_cooldown
                ),
                reason=(
                    f"{status_code}:{code}"
                ),
            )
            return

        if status_code == 429:
            self._mark_cooldown(
                model=model,
                status="rate_limited",
                seconds=(
                    self.rate_limit_cooldown
                ),
                reason=(
                    f"429:{code}"
                ),
            )
            return

        if (
            status_code == 408
            or status_code >= 500
        ):
            self._mark_cooldown(
                model=model,
                status=(
                    "temporarily_unavailable"
                ),
                seconds=(
                    self.transient_cooldown
                ),
                reason=(
                    f"{status_code}:{code}"
                ),
            )
            return

        raise AlibabaApplicationError(
            f"Unexpected Alibaba response "
            f"status {status_code}"
        )

    # ─────────────────────────────────
    # Main generation
    # ─────────────────────────────────

    async def generate_stream(
        self,
        user_context: str,
    ) -> AsyncIterator[str]:

        if not self.api_key:
            raise AlibabaNotConfiguredError(
                "Alibaba API key is not configured"
            )

        if not self.models:
            raise AlibabaNotConfiguredError(
                "No Alibaba models configured"
            )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_context,
            },
        ]

        client = self._get_client()

        last_error = None

        # Try Alibaba models in order
        for model in self.models:

            skip_reason = (
                self._should_skip(model)
            )

            if skip_reason:
                logger.info(
                    f"Alibaba model {model} "
                    f"skipped: {skip_reason}"
                )
                continue

            logger.info(
                f"LLM provider=Alibaba "
                f"model={model} attempting"
            )

            emitted = False
            full_response = []

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.3,
                "max_tokens": 1024,
            }

            try:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                ) as response:

                    if response.status_code >= 400:
                        body = (
                            await response.aread()
                        ).decode(
                            "utf-8",
                            errors="replace",
                        )

                        self._handle_http_error(
                            model=model,
                            status_code=(
                                response.status_code
                            ),
                            body=body,
                        )

                        last_error = (
                            f"HTTP "
                            f"{response.status_code}"
                        )

                        continue

                    async for line in (
                        response.aiter_lines()
                    ):
                        line = line.strip()

                        if not line:
                            continue

                        if not line.startswith(
                            "data:"
                        ):
                            continue

                        data = line[5:].strip()

                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(
                                data
                            )
                        except json.JSONDecodeError:
                            continue

                        choices = event.get(
                            "choices",
                            [],
                        )

                        if not choices:
                            continue

                        delta = (
                            choices[0]
                            .get("delta", {})
                        )

                        text = delta.get(
                            "content",
                            "",
                        )

                        if isinstance(text, list):
                            text = "".join(
                                part.get(
                                    "text",
                                    "",
                                )
                                for part in text
                                if isinstance(
                                    part,
                                    dict,
                                )
                            )

                        if not isinstance(
                            text,
                            str,
                        ):
                            continue

                        if not text:
                            continue

                        emitted = True

                        full_response.append(
                            text
                        )

                        yield text

                if emitted:
                    self._states[
                        model
                    ] = ModelState()

                    final_text = "".join(
                        full_response
                    ).strip()

                    logger.info(
                        f"LLM provider=Alibaba "
                        f"model={model} success "
                        f"chars={len(final_text)}"
                    )

                    return

                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason="empty_response",
                )

                last_error = (
                    "empty response"
                )

            except (
                AlibabaApplicationError,
                AlibabaProviderUnavailableError,
            ):
                raise

            except httpx.TimeoutException:
                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason="timeout",
                )

                logger.warning(
                    f"Alibaba model {model} "
                    f"timed out after "
                    f"{self.timeout_seconds}s"
                )

                last_error = "timeout"

            except httpx.RequestError as exc:
                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason=(
                        type(exc).__name__
                    ),
                )

                last_error = (
                    type(exc).__name__
                )

        raise AllAlibabaModelsFailed(
            "All configured Alibaba "
            "models failed"
            + (
                f": {last_error}"
                if last_error
                else ""
            )
        )

    async def generate(
        self,
        user_context: str,
    ) -> str:

        chunks = []

        async for chunk in (
            self.generate_stream(
                user_context
            )
        ):
            chunks.append(chunk)

        return "".join(chunks).strip()


alibaba_service = AlibabaService()