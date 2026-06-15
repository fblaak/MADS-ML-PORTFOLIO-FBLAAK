"""Helper for consistent modellogs location for the RNN assignment."""
from pathlib import Path

DEFAULT_LOGDIR = Path('3-hypertuning-rnn') / 'modellogs'

def ensure_logdir():
    DEFAULT_LOGDIR.mkdir(parents=True, exist_ok=True)
    return str(DEFAULT_LOGDIR)

def get_logdir():
    return str(DEFAULT_LOGDIR)
