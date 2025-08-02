
# Output helper for consistent logging across chained commands
import sys
import time


class OutputHelper:
    """Helper class to ensure consistent output in chained commands"""

    def __init__(self):
        # Force unbuffered output
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    def print_flushed(self, message, prefix="", force_flush=True):
        """Print with guaranteed flush"""
        full_message = f"{prefix}{message}"
        print(full_message, flush=force_flush)
        if force_flush:
            sys.stdout.flush()
            sys.stderr.flush()

    def log_with_timestamp(self, message, prefix="📝 "):
        """Log with timestamp for debugging"""
        timestamp = time.strftime("%H:%M:%S")
        self.print_flushed(f"[{timestamp}] {message}", prefix)

    def ensure_visibility(self, important_message):
        """Ensure critical messages are visible"""
        border = "=" * 50
        self.print_flushed(border)
        self.print_flushed(important_message, "🔥 IMPORTANT: ")
        self.print_flushed(border)

# Global instance for easy use
output_helper = OutputHelper()
