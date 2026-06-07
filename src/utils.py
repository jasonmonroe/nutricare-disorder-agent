# src/utils.py

# +-------------------+
# |     UTILITIES     |
# +-------------------+

# Python Libraries
from datetime import UTC, datetime
import os
import random
import time

# Local Libraries
from src.config import (
    AI_ROLE, 
    APP_TITLE, 
    GROQ_API_KEY, 
    HF_TOKEN, 
    LLAMA_KEY, 
    MAX_RUN_ID, 
    MEM0_API_KEY, 
    MIN_RUN_ID, 
    MSEC, 
    OPENAI_API_BASE, 
    OPENAI_API_KEY, 
    SECS_IN_MIN
)

def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return str(random.randint(MIN_RUN_ID, MAX_RUN_ID))


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()

def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"

def show_timer(start_time_int: float) -> None:
    print(f"⌚ Run Time: {get_time(start_time_int)}")

def show_banner(title: str, section: str = '') -> None:
    """Prints a stylized banner for console readability."""
    padding = 4
    strlen = len(title) + padding
    line = '+-' + '-' * strlen + '-+'

    print('')
    print(line)
    print('|  ' + title.upper() + '  |')
    print(line)

    if section:
        print('| ' + section)

    print('')

def show_title_banner() -> str:
    return f"""
        +-------------------------------------+
        |{APP_TITLE:^35}|
        |{AI_ROLE:^35}|
        +-------------------------------------+"""


def set_os_environ():
    # --- Environment Keys ---
    # Note: This line is for WRITING (or modifying) a variable within the Python process's environment.
    # Set the cleaned value back into the environment for libraries like LangChain to find
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY.strip()
    os.environ["LLAMA_KEY"] = LLAMA_KEY.strip()
    os.environ["MEM0_API_KEY"] = MEM0_API_KEY.strip()
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE.strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()
    os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
    # --- Environment Keys ---


    # --- HELPER FUNCTIONS
def show_datetime() -> str:
    now_utc = datetime.now(UTC)

    return now_utc.strftime("%b %d %Y %I:%M:%S %p %Z")

def handle_rate_limit_error(e, subject: str, current_sleep_time: int) -> tuple[int, bool]:
    """
    Checks for a Rate Limit Error (429), calculates a new sleep time,
    and returns the new sleep time and a flag indicating the hit.
    """
    i= 0
    print(f"{i}) Exception invoking a response for {subject}! Error: {e}")

    rate_limit_hit = False
    new_sleep_time = current_sleep_time

    if "Error code: 429" in str(e):
        # Increase sleep time by 15%
        rate_limit_hit = True
        new_sleep_time = current_sleep_time + round(current_sleep_time * 0.15)
        print(f"FATAL: Rate limit hit. Updating sleep time from {current_sleep_time} to {new_sleep_time} seconds...")
        if new_sleep_time > SECS_IN_MIN:
            new_sleep_time = SECS_IN_MIN

    return new_sleep_time, rate_limit_hit
