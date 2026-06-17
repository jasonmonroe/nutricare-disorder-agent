# src/utils.py

# +-------------------+
# |     UTILITIES     |
# +-------------------+

# Python Libraries
from datetime import UTC, datetime
import os
import random
import sys
import time

# Local Libraries
from src.constants import (
    AI_ROLE,
    APP_TITLE,
    AGENT_EXIT_CMDS,
    CHROMA_TELEMETRY_DISABLED,
    GROQ_API_KEY,
    HF_TOKEN,
    I_BOT,
    I_FLAG,
    I_HANDSHAKE,
    I_TIMER,
    LLAMA_KEY,
    RATE_LIMIT_TIME,
    RUN_MAX_ID,
    MEM0_API_KEY,
    RUN_MIN_ID,
    MSEC,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    SECS_IN_MIN,
    CHROMA_VECTORS_DIR,
    LLAMA_MODEL, RATE_LIMIT_RESP_CODE
)

# --- HELPER FUNCTIONS --- #

def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return str(random.randint(RUN_MIN_ID, RUN_MAX_ID))


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float, end_time_float: float | None = None) -> str:
    
    if end_time_float is None:
        end_time_float = time.time()

    diff = abs(end_time_float - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"{I_TIMER} Run Time: {get_time(start_time_int)}")


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


def show_models():
    print('+-------------------------------------+')
    print('| MODELS')
    print(f'| LLAMA_MODEL = {LLAMA_MODEL}')
    print(f'| OPENAI_EMBEDDING_MODEL = {OPENAI_EMBEDDING_MODEL}')
    print(f'| OPENAI_MODEL = {OPENAI_MODEL}')
    print('+-------------------------------------+')


def show_title_banner() -> None:
    print('+-------------------------------------+')
    print('|                                     |')
    print(f'|      {APP_TITLE}       |')
    print('|                                     |')
    print('+-------------------------------------+')
    print(f'|            {I_BOT} An AI Agent           |')
    print('+-------------------------------------+')
    print(f'\n{I_HANDSHAKE} You are a {AI_ROLE}. {I_HANDSHAKE}\n')

    show_models()


def show_ai_agent_banner() -> None:
    print('\n+--------------------------------------------------------------+')
    print(f'|        {I_BOT} SMART NUTRITION DISORDER SPECIALIST BOT {I_BOT}         |')
    print('+--------------------------------------------------------------+')
    print('| Welcome! I\'m your dedicated AI Nutrition Agent.              |')
    print('| Ask me anything about nutrition disorders, including their   |')
    print('| symptoms,causes, treatments, or preventative measures. I am  |')
    print('| here to assist with your health-related questions.           |')
    print('|                                                              |')
    print('+--------------------------------------------------------------+')
    print(f'| Type "{", ".join(AGENT_EXIT_CMDS)}" to end the conversation.              |')
    print('+--------------------------------------------------------------+\n')


def set_os_environ() -> None:
    # --- Environment Keys ---
    # Note: This line is for WRITING (or modifying) a variable within the Python process's environment.
    # Set the cleaned value back into the environment for libraries like LangChain to find
    os.environ["HF_TOKEN"] = str(HF_TOKEN).strip()
    os.environ["GROQ_API_KEY"] = str(GROQ_API_KEY).strip()
    os.environ["LLAMA_KEY"] = str(LLAMA_KEY).strip()
    os.environ["LLAMA_MODEL"] = str(LLAMA_MODEL).strip()
    os.environ["MEM0_API_KEY"] = str(MEM0_API_KEY).strip()
    os.environ["OPENAI_API_BASE"] = str(OPENAI_API_BASE).strip()
    os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY).strip()
    os.environ["OPENAI_EMBEDDING_MODEL"] = str(OPENAI_EMBEDDING_MODEL).strip()
    os.environ["OPENAI_MODEL"] = str(OPENAI_MODEL).strip()
    os.environ["CHROMA_TELEMETRY_DISABLED"] = CHROMA_TELEMETRY_DISABLED
    # --- Environment Keys ---


def show_datetime() -> str:
    now_utc = datetime.now(UTC)

    return now_utc.strftime("%b %d %Y %I:%M:%S %p %Z")


def handle_rate_limit_error(e, subject: str, current_sleep_time: int | float, i: int) -> tuple[int | float, bool]:
    """
    Checks for a Rate Limit Error (429), calculates a new sleep time, and returns the new sleep time and a flag
    indicating the hit.

    :param e:
    :param subject:
    :param current_sleep_time:
    :param i:
    :return:
    """
    rate_limit_hit = False
    new_sleep_time = current_sleep_time
    error_msg = str(e)

    if RATE_LIMIT_RESP_CODE in error_msg or "rate_limit_exceeded" in error_msg.lower():
        rate_limit_hit = True
        new_sleep_time = min(current_sleep_time * 2, SECS_IN_MIN)  # double it, cap at 60s
        print(f"{I_FLAG} Rate limit hit. Backing off from {current_sleep_time}s → {new_sleep_time}s...")
    else:
        print(f"{I_FLAG} {i}) Non-RateLimit Exception for {subject}: {error_msg}")

    return new_sleep_time, rate_limit_hit


def is_jupyter() -> bool:
    """
    Detects if the code is currently running inside a Jupyter Notebook
    or a standard terminal Python script.
    """
    # 1. Check if 'IPython' is even loaded in memory
    if 'IPython' not in sys.modules:
        return False
        
    try:
        from IPython import get_ipython
        # 2. Extract the name of the active shell class
        shell = get_ipython().__class__.__name__
        
        # 'ZMQInteractiveShell' corresponds to Jupyter Notebooks / JupyterLab
        if shell == 'ZMQInteractiveShell':
            return True
        # 'TerminalInteractiveShell' corresponds to the basic ipython terminal command
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
            
    except NameError:
        return False


def get_new_sleep_time() -> float:
        # Used in child classes.
        return RATE_LIMIT_TIME + random.uniform(2.0, 7.0)


def premium_model_tier() -> bool:
    # Are we using free tier models are expensive ones
    return LLAMA_MODEL == 'meta-llama/llama-guard-4-12b' and OPENAI_EMBEDDING_MODEL == 'text-embedding-3-small' and OPENAI_MODEL == 'gpt-4o-mini':


def format_dir(path: str) -> str:
    """
    Format persist directory.

    :param path:
    :return:
    """
    print(f'format_dir(./{CHROMA_VECTORS_DIR}/{path}_db)\n')
    return f"./{CHROMA_VECTORS_DIR}/{path}_db"
