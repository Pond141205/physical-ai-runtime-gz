import os
import time

from google import genai
from google.genai.errors import ServerError


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


def create_chat(client, model):
    return client.chats.create(
        model=model,
        config={"system_instruction": SYSTEM},
    )


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to start the Gemini chat.")

    client = genai.Client(api_key=api_key)
    current_model_index = 0
    chat = create_chat(client, MODELS[current_model_index])

    print("Physical AI Gemini Chat")
    print("พิมพ์ exit เพื่อออก")
    print("model:", MODELS[current_model_index])
    print()

    while True:
        user = input("YOU > ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        answered = False
        for model_index in range(current_model_index, len(MODELS)):
            model = MODELS[model_index]
            if model_index != current_model_index:
                print("\nSwitching model ->", model)
                chat = create_chat(client, model)
                current_model_index = model_index

            try:
                response = chat.send_message(user)
                print("\nAI >", response.text.strip())
                print("\n[model:", model + "]\n")
                answered = True
                break
            except ServerError as exc:
                if "503" in str(exc):
                    print("\n", model, "temporarily unavailable.", sep="")
                    time.sleep(2)
                    continue
                raise
            except Exception as exc:
                print("\nERROR:", type(exc).__name__, str(exc), "\n")
                break

        if not answered:
            print("\nAll configured Gemini models are currently unavailable.\n")


if __name__ == "__main__":
    main()
