# utils/logger.py
import time

class Logger:
    """Simple logger for time-stamped informational and error messages."""
    def __init__(self, prefix=""):
        self.prefix = prefix

    def info(self, msg):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t}] {self.prefix}{msg}")

    def error(self, msg):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t}] ERROR: {self.prefix}{msg}")