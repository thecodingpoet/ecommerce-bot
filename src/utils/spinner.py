"""
Spinner utility for CLI loading animations.
"""

import itertools
import sys
import threading
import time


class Spinner:
    """A simple loading spinner."""

    def __init__(self, message: str = "Processing"):
        self.spinner = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        """Run the spinner animation."""
        while self.running:
            sys.stdout.write(f"\r{next(self.spinner)} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
        sys.stdout.flush()
