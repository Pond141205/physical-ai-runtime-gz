import os
import time

from google import genai
from google.genai.errors import ServerError


client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

SYSTEM = """
You are the AI reasoning layer for a Physical AI robotics project.

You may discuss, reason, debug, and help design the system.

When later connected to RobotRuntime:
- choose semantic actions
- do not directly command joints
- do not invent unsafe actuator commands
- let the runtime/controller handle low-level motion
""".strip()


def create_chat(model):
    return client.chats.create(
        model=model,
        config={
            "system_instruction": SYSTEM,
        },
    )


current_model_index = 0
chat = create_chat(
    MODELS[current_model_index]
)

print("Physical AI Gemini Chat")
print("พิมพ์ exit เพื่อออก")
print("model:", MODELS[current_model_index])
print()

while True:

    user = input("YOU > ").strip()

    if user.lower() in {
        "exit",
        "quit",
    }:
        break

    answered = False

    for model_index in range(
        current_model_index,
        len(MODELS)
    ):
        model = MODELS[model_index]

        if model_index != current_model_index:
            print(
                "\nSwitching model ->",
                model
            )

            chat = create_chat(model)
            current_model_index = model_index

        try:
            response = chat.send_message(
                user
            )

            print(
                "\nAI >",
                response.text.strip()
            )

            print(
                "\n[model:",
                model + "]\n"
            )

            answered = True
            break

        except ServerError as e:

            if "503" in str(e):
                print(
                    "\n",
                    model,
                    "temporarily unavailable.",
                    sep=""
                )

                time.sleep(2)
                continue

            raise

        except Exception as e:
            print(
                "\nERROR:",
                type(e).__name__,
                str(e),
                "\n"
            )
            break

    if not answered:
        print(
            "\nAll configured Gemini models "
            "are currently unavailable.\n"
        )
