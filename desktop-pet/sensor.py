"""Perception Daemon for AI Desktop Pet (macOS).

Monitors:
1. Frontmost active application and window title.
2. Focused work duration (fatigue detection).
3. Clipboard content for errors, tracebacks, or code snippets.
"""

import subprocess
import time
from typing import Optional, Dict, Any
from PyQt6.QtCore import QThread, pyqtSignal


def get_frontmost_app_macos() -> str:
    """Retrieve the frontmost application name using AppleScript."""
    script = 'tell application "System Events" to get name of first process whose frontmost is true'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def get_frontmost_window_title_macos() -> str:
    """Retrieve the active window title of the frontmost application."""
    script = 'tell application "System Events" to get name of window 1 of (first process whose frontmost is true)'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_clipboard_text_macos() -> str:
    """Read current system clipboard text via pbpaste."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=0.5,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""


class SystemSensor(QThread):
    """Background polling thread that detects context events and emits signals."""

    # Emits context dictionary: {app, window_title, duration, event, clipboard_error}
    context_updated = pyqtSignal(dict)

    def __init__(self, poll_interval_sec: float = 2.0):
        super().__init__()
        self.poll_interval = poll_interval_sec
        self.running = True
        self.current_app = ""
        self.app_start_time = time.time()
        self.last_clipboard = ""

    def run(self):
        while self.running:
            try:
                front_app = get_frontmost_app_macos()
                window_title = get_frontmost_window_title_macos()
                now = time.time()

                # App switch detection
                app_switched = False
                if front_app and front_app != self.current_app:
                    self.current_app = front_app
                    self.app_start_time = now
                    app_switched = True

                duration = int(now - self.app_start_time)

                # Clipboard error detection
                clip_text = get_clipboard_text_macos()
                clipboard_error = None
                if clip_text and clip_text != self.last_clipboard:
                    self.last_clipboard = clip_text
                    lower_clip = clip_text.lower()
                    error_keywords = [
                        "traceback", "syntaxerror", "typeerror", "valueerror",
                        "indexerror", "keyerror", "segmentation fault",
                        "nullpointerexception", "panic:", "fatal error",
                        "exception in thread", "failed with exit code",
                    ]
                    if any(kw in lower_clip for kw in error_keywords):
                        # Extract first 2-3 lines of error
                        lines = [line.strip() for line in clip_text.strip().split("\n") if line.strip()]
                        clipboard_error = lines[-1] if lines else "Exception occurred"
                        if len(clipboard_error) > 80:
                            clipboard_error = clipboard_error[:77] + "..."

                # Build event payload
                event_type = "poll"
                if app_switched:
                    event_type = "app_switch"
                elif clipboard_error:
                    event_type = "clipboard_error"

                payload = {
                    "app": front_app,
                    "window_title": window_title,
                    "duration_seconds": duration,
                    "event": event_type,
                    "clipboard_error": clipboard_error,
                }

                self.context_updated.emit(payload)

            except Exception:
                pass

            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False
        self.wait()
