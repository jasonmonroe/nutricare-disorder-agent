# src/utils.py

# +-------------------+
# |     UTILITIES     |
# +-------------------+

# Python Libraries
from datetime import UTC, datetime
import os
import random
import sys
import textwrap
import time
import uuid

# Local Libraries
from src.constants import (
    AGENT_EXIT_CMDS,
    AI_ROLE,
    APP_TITLE,
    CHROMA_TELEMETRY_DISABLED,
    GROQ_API_KEY,
    HF_TOKEN,
    I_BOT,
    I_FLAG,
    I_HANDSHAKE,
    I_TIMER,
    LLAMA_KEY,
    LLAMA_MODEL,
    MEM0_API_KEY,
    MSEC,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    PEP8_LINE_LEN,
    RATE_LIMIT_RESP_CODE,
    RUN_MAX_ID,
    RUN_MIN_ID,
    SECS_IN_MIN
)
from src.model_config import config

# --- HELPER FUNCTIONS --- #

def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return str(random.randint(RUN_MIN_ID, RUN_MAX_ID))


def gen_uuid() -> str:
    return uuid.uuid4().hex.lower()


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


def _make_top_btm_line() -> str:
    open_close_len = 2 # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    return '+' + ('-' * max_line_len) + '+'


def _create_title_banner(text: str, center_text: bool=True) -> None:
    open_close_len = 4 # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    # Trim off any chars after limit plus two spaces for blanks
    text = text[0: max_line_len - open_close_len]
    text_len = len(text)
    padding_len = max_line_len - text_len    
   
    if center_text:
        # If uneven padding add an extra length for the right side
        extra_len = 0 if padding_len % 2 == 0 else 1
      
        padding_len = padding_len // 2
        title_line = "| " + (' ' * padding_len) + text + (' ' * (padding_len + extra_len)) + " |"
    
    else:
        # Remove last two characters to account for open/close spacing
        title_line = "| " + text + (' ' * padding_len) + " |"
        
    top_btm_line = _make_top_btm_line()

    # Print title banner
    print("\n")
    print(top_btm_line)
    print(title_line)
    print(top_btm_line)


def _create_subtitle_banner(text: str | list, center_text: bool=False) -> None:
    # Reconstructs the guard to safely catch wrong types OR empty values
    if not isinstance(text, (str, list)) or not text:
        print('return None')
        return None
        
    open_close_len = 4 # open close of char `+` or `|` plus space
    max_line_len = PEP8_LINE_LEN - open_close_len 
    wrapped_lines = []

    if isinstance(text, list):
        wrapped_lines = text

    elif isinstance(text, str):
        wrapped_lines = textwrap.wrap(text, width=max_line_len)

    # Now that the data is a list format it for display.
    for line in wrapped_lines:
        if "\n" in line:
            line = ""

        line_len = len(line)
    
        padding_len = max_line_len - line_len
      
        if center_text:    
            extra_len = 0 if padding_len % 2 == 0 else 1
            padding_len = padding_len // 2
            padded_line = "| " + (' ' * padding_len) + line + (' ' * (padding_len + extra_len)) + " |"
        else:
            padded_line = "| " + line + (' ' * padding_len) + " |"
  
        print(padded_line)
       
    # Close the subtitle
    if len(wrapped_lines) > 0:
        print(_make_top_btm_line())

    return None

def show_banner(title: str, subtitle: str | list | None="", center_title_text: bool=True, center_subtitle_text: bool=False) -> None:
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)


def show_title_banner() -> None:
    _create_title_banner(APP_TITLE)
    print(f'\n{I_HANDSHAKE} You are a {AI_ROLE}. {I_HANDSHAKE}\n')


def show_model_banner() -> None:
    title = f'⚙️ ({config.VERSION.upper()}) MODELS ⚙️'
    subtitle = [f'LLAMA_MODEL: {LLAMA_MODEL}', f'OPENAI_EMBEDDING_MODEL: {OPENAI_EMBEDDING_MODEL}', f'OPENAI_MODEL: {OPENAI_MODEL}']
    
    _create_title_banner(title)
    _create_subtitle_banner(subtitle)
    

def show_ai_agent_banner() -> None:
    title = f'{I_BOT} SMART NUTRITION DISORDER SPECIALIST BOT {I_BOT}'
    subtitle = 'Welcome! I\'m your dedicated AI Nutrition Agent.\nAsk me anything about nutrition disorders, including their symptoms,\ncauses, treatments, or preventative measures. I am here to assist with your\nhealth-related questions.'
 
    _create_title_banner(title)
    _create_subtitle_banner(subtitle)

    print(f'Type "{", ".join(AGENT_EXIT_CMDS)}" to end the conversation.\n')


def set_os_environ() -> None:
    # --- Environment Keys ---
    # ℹ️ Note: This line is for WRITING (or modifying) a variable within the Python process's environment.
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
    # Check if 'IPython' is even loaded in memory
    if 'IPython' not in sys.modules:
        return False
        
    try:
        from IPython import get_ipython
        # Extract the name of the active shell class
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
    return config.RATE_LIMIT_TIME + random.uniform(2.0, 7.0)


def format_dir(path: str) -> str:
    """
    Format persist directory.

    :param path:
    :return:
    """
    print(f'format_dir(./{config.CHROMA_VECTORS_DIR}/{path}_db)\n')
    return f"./{config.CHROMA_VECTORS_DIR}/{path}_db"
