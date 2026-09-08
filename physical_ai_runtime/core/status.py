"""User-visible operational status reporting.

This module exposes system state only.
It must never expose private model reasoning / chain-of-thought.
"""


class RuntimeStatus:
    RESET = "\033[0m"
    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"

    @classmethod
    def _print(cls, color, label, message):
        print(
            f"{color}[{label}]{cls.RESET} "
            f"{message}",
            flush=True,
        )

    @classmethod
    def ai(cls, message):
        cls._print(
            cls.CYAN,
            "AI",
            message,
        )

    @classmethod
    def vision(cls, message):
        cls._print(
            cls.CYAN,
            "VISION",
            message,
        )

    @classmethod
    def perception(cls, message):
        cls._print(
            cls.CYAN,
            "PERCEPTION",
            message,
        )

    @classmethod
    def geometry(cls, message):
        cls._print(
            cls.CYAN,
            "GEOMETRY",
            message,
        )

    @classmethod
    def planner(cls, message):
        cls._print(
            cls.CYAN,
            "PLANNER",
            message,
        )

    @classmethod
    def safety(cls, message):
        cls._print(
            cls.CYAN,
            "SAFETY",
            message,
        )

    @classmethod
    def motion(cls, message):
        cls._print(
            cls.CYAN,
            "MOTION",
            message,
        )

    @classmethod
    def verify(cls, message):
        cls._print(
            cls.CYAN,
            "VERIFY",
            message,
        )

    @classmethod
    def ok(cls, message):
        cls._print(
            cls.GREEN,
            "OK",
            message,
        )

    @classmethod
    def info(cls, message):
        cls._print(
            cls.YELLOW,
            "INFO",
            message,
        )

    @classmethod
    def fail(cls, message):
        cls._print(
            cls.RED,
            "FAIL",
            message,
        )
