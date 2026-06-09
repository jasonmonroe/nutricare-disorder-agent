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
from app import LLAMA_MODEL
from src.config import (
    AI_ROLE,
    AI_TITLE, 
    APP_TITLE,
    EXIT_CMD, 
    GROQ_API_KEY, 
    HF_TOKEN,
    I_BOT, 
    I_FLAG,
    I_HANDSHAKE,
    LLAMA_KEY, 
    MAX_RUN_ID, 
    MEM0_API_KEY, 
    MIN_RUN_ID, 
    MSEC, 
    OPENAI_API_BASE, 
    OPENAI_API_KEY, 
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    SECS_IN_MIN,
    SLEEP_TIME_INC
)

def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return str(random.randint(MIN_RUN_ID, MAX_RUN_ID))


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()

def get_time(start_time_float: float, end_time_float: float=None) -> str:
    
    if end_time_float is None:
        end_time_float = time.time()

    diff = abs(end_time_float - start_time_float)
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
    print('+-------------------------------------+')
    print('|                                     |')
    print(f'|      {APP_TITLE}       |')
    print('|                                     |')
    print('+-------------------------------------+')
    print(f'# === {I_HANDSHAKE} You are a {AI_ROLE}. {I_HANDSHAKE} === #')
    print('+-------------------------------------+\n')
    
def show_ai_agent_banner() -> str:
    print('\n+--------------------------------------------------------------+')
    print(f'|        {I_BOT} SMART NUTRITION DISORDER SPECIALIST BOT {I_BOT}         |')
    print('+--------------------------------------------------------------+')
    print('| Welcome! I\'m your dedicated AI Nutrition Agent.              |')
    print('| Ask me anything about nutrition disorders, including their   |')
    print('| symptoms,causes, treatments, or preventative measures. I am  |')
    print('| here to  assist with your health-related questions.          |')
    print('|                                                              |')
    print('+--------------------------------------------------------------+')
    print(f'| Type "{EXIT_CMD}" to end the conversation.                         |')
    print('+--------------------------------------------------------------+\n')

def set_os_environ():
    # --- Environment Keys ---
    # Note: This line is for WRITING (or modifying) a variable within the Python process's environment.
    # Set the cleaned value back into the environment for libraries like LangChain to find
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY.strip()
    os.environ["LLAMA_KEY"] = LLAMA_KEY.strip()
    os.environ["LLAMA_MODEL"] = LLAMA_MODEL.strip()
    os.environ["MEM0_API_KEY"] = MEM0_API_KEY.strip()
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE.strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()
    os.environ["OPENAI_EMBEDDING_MODEL"] = OPENAI_EMBEDDING_MODEL.strip()
    os.environ["OPENAI_MODEL"] = OPENAI_MODEL.strip()
    os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
    # --- Environment Keys ---

# --- HELPER FUNCTIONS --- #
def show_datetime() -> str:
    now_utc = datetime.now(UTC)

    return now_utc.strftime("%b %d %Y %I:%M:%S %p %Z")

def handle_rate_limit_error(e, subject: str, current_sleep_time: int, i:int) -> tuple[int, bool]:
    """
    Checks for a Rate Limit Error (429), calculates a new sleep time,
    and returns the new sleep time and a flag indicating the hit.
    """
    print(f"{I_FLAG} {i}) Exception invoking a response for {subject}! Error: {e}")

    rate_limit_hit = False
    new_sleep_time = current_sleep_time

    if "Error code: 429" in str(e):
        # Increase sleep time by 15%
        rate_limit_hit = True
        new_sleep_time = current_sleep_time + round(current_sleep_time * SLEEP_TIME_INC)
        print(f"{I_FLAG} FATAL: Rate limit hit. Updating sleep time from {current_sleep_time} to {new_sleep_time} seconds...")
        if new_sleep_time > SECS_IN_MIN:
            new_sleep_time = SECS_IN_MIN

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
        # 'TerminalInteractiveShell' corresponds to the basic ipython terminal terminal command
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
            
    except NameError:
        return False
