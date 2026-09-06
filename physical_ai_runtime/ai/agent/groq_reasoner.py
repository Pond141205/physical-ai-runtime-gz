import json
import os

from groq import Groq


class GroqReasoner:
    """
    Cloud semantic reasoner.

    The model may only select semantic actions exposed by the runtime.
    It never generates joints, TCP coordinates, actuator values,
    or motion corrections.
    """

    def __init__(
        self,
        model="openai/gpt-oss-120b",
    ):
        api_key = os.environ.get(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY_NOT_SET"
            )

        self.model = model
        self.client = Groq(
            api_key=api_key
        )

    def __call__(
        self,
        context,
    ):
        available = context.get(
            "available_actions",
            []
        )

        if not available:
            raise RuntimeError(
                "NO_AVAILABLE_ACTIONS"
            )

        system_prompt = """
You are the semantic decision layer of a safety-first
Physical AI runtime.

You may ONLY choose one action from available_actions.

You MUST NOT:
- generate joint positions
- generate TCP XYZ coordinates
- generate quaternions
- generate motor commands
- generate correction distances
- bypass runtime safety checks

The robot runtime and motion planner decide HOW motion
is performed and verify collision safety.

Choose the safest semantically appropriate next action
based only on the supplied feedback, metrics, history,
and available actions.

If changing grasp strategy, strategy must be a semantic
strategy name, not a pose or numeric motion command.

Return only the structured response.
""".strip()

        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(available),
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
            "additionalProperties": False,
        }

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            default=str,
                        ),
                    },
                ],
                temperature=0.1,
                reasoning_effort="medium",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "semantic_action",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "GROQ_EMPTY_RESPONSE"
            )

        decision = json.loads(
            content
        )

        action = decision.get(
            "action"
        )

        if action not in available:
            raise RuntimeError(
                "GROQ_SELECTED_UNAVAILABLE_ACTION:"
                + str(action)
            )

        # Strategy is meaningful only when the semantic action
        # explicitly requests a grasp-strategy change.
        if action != "CHANGE_GRASP_STRATEGY":
            decision["strategy"] = ""

        else:
            strategy = str(
                decision.get("strategy", "")
            ).strip().lower()

            allowed_strategies = {
                "angled",
                "adjust_grasp_angle_and_approach",
            }

            if strategy not in allowed_strategies:
                raise RuntimeError(
                    "GROQ_SELECTED_UNSUPPORTED_STRATEGY:"
                    + strategy
                )

            decision["strategy"] = strategy

        return decision
