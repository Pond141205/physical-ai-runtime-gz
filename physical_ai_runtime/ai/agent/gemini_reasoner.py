import json
import os
import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError


class GeminiSemanticReasoner:
    """
    Cloud AI decision layer.

    Gemini chooses WHAT semantic action to perform.
    RobotRuntime / controller decides HOW to execute it.
    """

    def __init__(self, model=None):
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY_NOT_SET")

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = (
            model
            or os.environ.get(
                "PHYSICAL_AI_GEMINI_MODEL",
                "gemini-3.6-flash",
            )
        )

    @staticmethod
    def _quota_retry_seconds(error):
        text = str(error)

        marker = "Please retry in "
        if marker in text:
            tail = text.split(marker, 1)[1]
            value = tail.split("s", 1)[0]
            try:
                return float(value)
            except ValueError:
                pass

        marker = "'retryDelay': '"
        if marker in text:
            tail = text.split(marker, 1)[1]
            value = tail.split("s'", 1)[0]
            try:
                return float(value)
            except ValueError:
                pass

        return None

    def __call__(self, context):
        available = (
            context.get("available_actions")
            or []
        )

        if not available:
            return {
                "action": "ABORT",
                "reason": "No runtime actions are available.",
                "strategy": None,
            }

        system_instruction = """
You are the semantic decision layer of a Physical AI runtime.

You receive:
- semantic robot feedback
- perception/execution metrics
- task history
- actions currently allowed by the runtime

Choose WHAT the robot should do next.

You MUST:
- choose exactly one action from available_actions
- reason from the current feedback and history
- avoid repeating a recovery that is clearly not improving the state

You MUST NOT:
- invent unavailable actions
- command joints
- produce TCP coordinates
- produce motor commands
- produce correction distances
- bypass runtime safety
- assume a specific robot embodiment

The RobotRuntime and robot adapter determine HOW the action is executed.

If no strategy is needed, return an empty string for strategy.
""".strip()

        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": available,
                },
                "reason": {
                    "type": "string",
                },
                "strategy": {
                    "type": "string",
                },
            },
            "required": [
                "action",
                "reason",
                "strategy",
            ],
        }

        prompt = json.dumps(
            context,
            default=self._json_default,
            indent=2,
        )

        last_error = None
        server_attempt = 0
        quota_retries = 0

        max_server_attempts = 3
        max_quota_retries = 2

        while True:
            try:
                chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=
                    system_instruction,
                response_mime_type=
                    "application/json",
                response_schema=schema,
                temperature=0.2,
            ),
        )

                response = chat.send_message(
                    prompt
                )

                decision = json.loads(
                    response.text
                )

                action = decision.get(
                    "action"
                )

                if action not in available:
                    raise RuntimeError(
                        "GEMINI_SELECTED_UNAVAILABLE_ACTION: "
                        f"{action}"
                    )

                return decision

            except ClientError as e:
                text = str(e)
                code = getattr(e, "code", None)

                if (
                    code == 429
                    or "RESOURCE_EXHAUSTED" in text
                ):
                    delay = self._quota_retry_seconds(e)

                    if delay is None:
                        raise RuntimeError(
                            "GEMINI_QUOTA_DELAY_UNAVAILABLE"
                        ) from e

                    quota_retries += 1

                    if quota_retries > max_quota_retries:
                        raise RuntimeError(
                            "GEMINI_QUOTA_RETRY_EXHAUSTED"
                        ) from e

                    print("\nGEMINI QUOTA HOLD")
                    print("Robot motion: HOLD")
                    print(
                        "Retry after:",
                        delay + 1.0,
                        "seconds"
                    )

                    time.sleep(delay + 1.0)
                    continue

                raise RuntimeError(
                    "GEMINI_CLIENT_ERROR:"
                    + text
                ) from e

            except ServerError as e:
                last_error = e
                server_attempt += 1

                print(
                    "GEMINI SERVER ERROR "
                    f"(attempt {server_attempt}/"
                    f"{max_server_attempts})"
                )

                if (
                    server_attempt
                    >= max_server_attempts
                ):
                    break

                time.sleep(
                    2 ** (
                        server_attempt - 1
                    )
                )

        raise RuntimeError(
            "GEMINI_UNAVAILABLE"
        ) from last_error

    @staticmethod
    def _json_default(value):
        if hasattr(value, "tolist"):
            return value.tolist()

        return str(value)
