#!/usr/bin/env python3
"""AI Desktop Pet (Clippy 2.0) - A context-aware companion on your screen.

Built with pure Python and PyQt6.
Features:
- Transparent, frameless, always-on-top floating pet.
- Expressive procedural vector animations (blinking, breathing, sleeping, alerts).
- Real-time awareness of active macOS applications, focus time, and clipboard errors.
- Dynamic comic speech bubble with instant typewriter chat.
- 4 Switchable personalities (Clippy, Cyber Cat, Hacker, Zen Master).
- Pluggable local LLM support (Ollama / llama.cpp / vLLM) with zero-lag offline fallback.
"""

import sys
import os
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import (
    Qt, QPoint, QRectF, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QObject
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QCursor, QPainterPath, QLinearGradient
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QMenu, QGraphicsDropShadowEffect
)

from sensor import SystemSensor
from brain import BrainEngine, PERSONALITIES


CONFIG_PATH = Path.home() / ".ai_desktop_pet_config.json"


def make_window_always_on_top_macos(widget: QWidget):
    """Force macOS NSWindow to float above all apps, spaces, and full-screen windows."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util
        view_ptr = int(widget.winId())
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p

        sel_window = objc.sel_registerName(b'window')
        ns_window = objc.objc_msgSend(ctypes.c_void_p(view_ptr), sel_window)
        if not ns_window:
            return

        # Lift window level above all normal floating windows (1000 = kCGOverlayWindowLevelKey)
        sel_setLevel = objc.sel_registerName(b'setLevel:')
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        objc.objc_msgSend.restype = None
        objc.objc_msgSend(ns_window, sel_setLevel, 1000)

        # Set collection behavior: canJoinAllSpaces (1) | stationary (16) | fullScreenAuxiliary (256)
        sel_setCollectionBehavior = objc.sel_registerName(b'setCollectionBehavior:')
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        behavior = 1 | 16 | 256
        objc.objc_msgSend(ns_window, sel_setCollectionBehavior, behavior)
    except Exception:
        pass


class SpeechBubble(QWidget):
    """Comic speech bubble floating near the pet with interactive chat."""

    message_submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.full_text = ""
        self.displayed_text = ""
        self.char_index = 0

        # Typewriter animation timer
        self.type_timer = QTimer(self)
        self.type_timer.timeout.connect(self._type_next_char)

        # Auto-hide timer
        self.auto_hide_timer = QTimer(self)
        self.auto_hide_timer.setSingleShot(True)
        self.auto_hide_timer.timeout.connect(self.hide_bubble)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # Text label
        self.label = QLabel("...", self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(
            "color: #1a1a1a; font-size: 13px; font-weight: 500; line-height: 1.4;"
        )
        layout.addWidget(self.label)

        # Bottom row: interactive input box
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("和它说点什么 (按回车发送)...")
        self.input_field.setStyleSheet(
            """
            QLineEdit {
                background-color: #f3f3f0;
                border: 1px solid #d4d4cc;
                border-radius: 12px;
                padding: 4px 10px;
                font-size: 12px;
                color: #222222;
            }
            QLineEdit:focus {
                border: 1px solid #ff6600;
                background-color: #ffffff;
            }
            """
        )
        self.input_field.returnPressed.connect(self._on_submit)
        bottom_layout.addWidget(self.input_field)

        layout.addLayout(bottom_layout)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(0, 0, 0, 45))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        self.setFixedWidth(280)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height() - 10)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)

        # Little pointer triangle at the bottom left
        tail_start_x = 36
        tail = QPainterPath()
        tail.moveTo(tail_start_x, self.height() - 10)
        tail.lineTo(tail_start_x + 12, self.height())
        tail.lineTo(tail_start_x + 24, self.height() - 10)
        tail.closeSubpath()
        path = path.united(tail)

        # Fill with gentle white-cream gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(255, 255, 255, 248))
        gradient.setColorAt(1.0, QColor(248, 248, 242, 248))

        painter.fillPath(path, QBrush(gradient))
        painter.strokePath(path, QPen(QColor(210, 210, 200, 200), 1.2))

    def show_message(self, text: str, auto_hide_sec: float = 8.0):
        self.full_text = text
        self.displayed_text = ""
        self.char_index = 0
        self.label.setText("")

        self.adjustSize()
        self.show()
        self.raise_()
        make_window_always_on_top_macos(self)

        self.type_timer.start(25)  # 25ms per character for smooth typing
        if auto_hide_sec > 0:
            self.auto_hide_timer.start(int(auto_hide_sec * 1000))

    def _type_next_char(self):
        if self.char_index < len(self.full_text):
            self.displayed_text += self.full_text[self.char_index]
            self.char_index += 1
            self.label.setText(self.displayed_text)
            self.adjustSize()
        else:
            self.type_timer.stop()

    def _on_submit(self):
        msg = self.input_field.text().strip()
        if msg:
            self.input_field.clear()
            self.message_submitted.emit(msg)

    def hide_bubble(self):
        self.type_timer.stop()
        self.auto_hide_timer.stop()
        self.hide()


class DesktopPet(QWidget):
    """The interactive, animated desktop pet sprite."""

    def __init__(self):
        super().__init__()

        # Window Flags: Frameless, Always on Top, Tool (floating panel on macOS)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Load configuration
        self.config = self._load_config()
        initial_personality = self.config.get("personality", "clippy")

        # Modules
        self.brain = BrainEngine(personality=initial_personality)
        self.sensor = SystemSensor(poll_interval_sec=2.5)

        # Animation states
        self.state = "idle"  # idle, working, alert, sleeping, bounce
        self.anim_tick = 0
        self.blink_state = 0  # 0: open, 1: half, 2: closed
        self.bounce_offset = 0.0

        # Mouse dragging state
        self.drag_position = None

        # Setup Speech Bubble
        self.bubble = SpeechBubble()
        self.bubble.message_submitted.connect(self._handle_user_message)

        # Animation Timers
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.start(50)  # 20 FPS

        # Periodic thought timer
        self.thought_timer = QTimer(self)
        self.thought_timer.timeout.connect(self._trigger_periodic_thought)
        self.thought_timer.start(45000)  # Thought every 45s

        # Connect sensor
        self.sensor.context_updated.connect(self._on_context_updated)
        self.sensor.start()

        self.last_context = {}

        # Set size and initial position
        self.resize(110, 110)

        # Calculate smart default position based on primary screen available geometry
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            default_x = avail.x() + avail.width() - 190
            default_y = avail.y() + avail.height() - 210
        else:
            default_x, default_y = 1200, 750

        init_x = self.config.get("x", default_x)
        init_y = self.config.get("y", default_y)
        self.move(init_x, init_y)
        self._update_bubble_position()

        # Say hello immediately on launch
        QTimer.singleShot(200, self._say_welcome)

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"personality": "clippy"}

    def _save_config(self):
        data = {
            "x": self.x(),
            "y": self.y(),
            "personality": self.brain.personality,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _say_welcome(self):
        info = self.brain.get_personality_info()
        welcome_msgs = {
            "clippy": "嗨！我是你的桌面小助手 Clippy，今天我们做点什么精彩的？",
            "cat": "喵呜~ 本猫刚刚巡视完你的屏幕，记得多敲代码多喂鱼！",
            "hacker": "系统初始化完成。终端已锁定，准备秒杀一切 Bug。",
            "zen": "心如止水，身如琉璃。愿你今日专注而平静。",
        }
        msg = welcome_msgs.get(self.brain.personality, "你好呀！我是你的桌面精灵！")
        self.say(msg)

    def showEvent(self, event):
        super().showEvent(event)
        make_window_always_on_top_macos(self)

    def say(self, text: str, duration_sec: float = 8.0):
        self._update_bubble_position()
        self.bubble.show_message(text, auto_hide_sec=duration_sec)
        make_window_always_on_top_macos(self)

    def _update_bubble_position(self):
        # Position bubble above or to the left of the pet
        bubble_x = self.x() - 120
        bubble_y = self.y() - self.bubble.height() - 8
        if bubble_y < 40:
            bubble_y = self.y() + self.height() + 10
        if bubble_x < 20:
            bubble_x = 20
        self.bubble.move(bubble_x, bubble_y)

    def _update_animation(self):
        self.anim_tick += 1

        # Periodic blinking logic
        if self.anim_tick % 60 == 0:
            self.blink_state = 1
        elif self.anim_tick % 60 == 2:
            self.blink_state = 2
        elif self.anim_tick % 60 == 4:
            self.blink_state = 1
        elif self.anim_tick % 60 == 6:
            self.blink_state = 0

        # Subtle floating bounce
        if self.state != "sleeping":
            self.bounce_offset = math.sin(self.anim_tick * 0.1) * 3.0
        else:
            self.bounce_offset = 0.0

        self.update()

    def _on_context_updated(self, context: dict):
        self.last_context = context
        event = context.get("event")

        # If clipboard caught a code exception, trigger alert!
        if event == "clipboard_error":
            self.state = "alert"
            remark = self.brain.think(context)
            self.say(remark, duration_sec=10.0)

        # If user switched app
        elif event == "app_switch":
            app = context.get("app", "").lower()
            if any(term in app for term in ["code", "xcode", "cursor", "pycharm", "neovim", "ghostty", "terminal"]):
                self.state = "working"
            else:
                self.state = "idle"

    def _trigger_periodic_thought(self):
        if not self.bubble.isVisible():
            remark = self.brain.think(self.last_context)
            self.say(remark, duration_sec=7.0)

    def _handle_user_message(self, message: str):
        reply = self.brain.chat(message)
        self.say(reply, duration_sec=9.0)

    # -------------------------------------------------------------
    # Procedural Vector Art Painter
    # -------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        personality = self.brain.personality
        cx = self.width() / 2.0
        cy = (self.height() / 2.0) + self.bounce_offset

        if personality == "clippy":
            self._draw_clippy(painter, cx, cy)
        elif personality == "cat":
            self._draw_cat(painter, cx, cy)
        elif personality == "hacker":
            self._draw_hacker(painter, cx, cy)
        elif personality == "zen":
            self._draw_zen(painter, cx, cy)

    def _draw_clippy(self, painter: QPainter, cx: float, cy: float):
        """Draw classic metallic paperclip with googly animated eyes."""
        # Paperclip metallic gradient wire
        pen = QPen(QColor(180, 185, 195), 5.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        path = QPainterPath()
        path.moveTo(cx - 14, cy + 24)
        path.lineTo(cx - 14, cy - 20)
        path.arcTo(cx - 14, cy - 36, 30, 30, 180, -180)
        path.lineTo(cx + 16, cy + 18)
        path.arcTo(cx - 8, cy + 8, 24, 24, 0, -180)
        path.lineTo(cx - 8, cy - 10)
        path.arcTo(cx - 8, cy - 20, 16, 16, 180, -180)
        path.lineTo(cx + 8, cy + 12)
        painter.drawPath(path)

        # Highlight reflection on wire
        pen_glow = QPen(QColor(245, 248, 255, 180), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_glow)
        painter.drawPath(path)

        # Eyes
        eye_y = cy - 10
        left_eye_x = cx - 8
        right_eye_x = cx + 8

        # Eye whites
        painter.setPen(QPen(QColor(40, 40, 40), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPoint(int(left_eye_x), int(eye_y)), 7, 9)
        painter.drawEllipse(QPoint(int(right_eye_x), int(eye_y)), 7, 9)

        # Pupils (animated blink)
        if self.blink_state == 0 and self.state != "sleeping":
            painter.setBrush(QBrush(QColor(20, 20, 20)))
            # Pupils look towards mouse or forward
            pupil_offset_x = math.sin(self.anim_tick * 0.05) * 2.0
            painter.drawEllipse(QPoint(int(left_eye_x + pupil_offset_x), int(eye_y)), 3, 4)
            painter.drawEllipse(QPoint(int(right_eye_x + pupil_offset_x), int(eye_y)), 3, 4)

            # Tiny white specular reflection
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QPoint(int(left_eye_x + pupil_offset_x - 1), int(eye_y - 2)), 1, 1)
            painter.drawEllipse(QPoint(int(right_eye_x + pupil_offset_x - 1), int(eye_y - 2)), 1, 1)
        else:
            # Sleeping or blinking slit
            painter.setPen(QPen(QColor(20, 20, 20), 2.0))
            painter.drawLine(int(left_eye_x - 4), int(eye_y), int(left_eye_x + 4), int(eye_y))
            painter.drawLine(int(right_eye_x - 4), int(eye_y), int(right_eye_x + 4), int(eye_y))

        # Eyebrows
        painter.setPen(QPen(QColor(50, 50, 50), 1.8))
        if self.state == "alert":
            painter.drawLine(int(left_eye_x - 5), int(eye_y - 13), int(left_eye_x + 3), int(eye_y - 15))
            painter.drawLine(int(right_eye_x - 3), int(eye_y - 15), int(right_eye_x + 5), int(eye_y - 13))
        else:
            painter.drawLine(int(left_eye_x - 4), int(eye_y - 12), int(left_eye_x + 4), int(eye_y - 12))
            painter.drawLine(int(right_eye_x - 4), int(eye_y - 12), int(right_eye_x + 4), int(eye_y - 12))

    def _draw_cat(self, painter: QPainter, cx: float, cy: float):
        """Draw cute pixel/chibi cat with twitching ears."""
        # Cat head base
        painter.setPen(QPen(QColor(50, 45, 40), 1.8))
        painter.setBrush(QBrush(QColor(255, 245, 230)))
        painter.drawEllipse(QPoint(int(cx), int(cy)), 24, 20)

        # Ears
        ear_twitch = math.sin(self.anim_tick * 0.15) * 2.0
        # Left ear
        left_ear = QPainterPath()
        left_ear.moveTo(cx - 18, cy - 10)
        left_ear.lineTo(cx - 24 + ear_twitch, cy - 28)
        left_ear.lineTo(cx - 6, cy - 18)
        left_ear.closeSubpath()
        painter.drawPath(left_ear)

        # Right ear
        right_ear = QPainterPath()
        right_ear.moveTo(cx + 6, cy - 18)
        right_ear.lineTo(cx + 24 - ear_twitch, cy - 28)
        right_ear.lineTo(cx + 18, cy - 10)
        right_ear.closeSubpath()
        painter.drawPath(right_ear)

        # Inner ear pink
        painter.setBrush(QBrush(QColor(255, 180, 190)))
        painter.setPen(Qt.PenStyle.NoPen)
        inner_l = QPainterPath()
        inner_l.moveTo(cx - 16, cy - 12)
        inner_l.lineTo(cx - 22 + ear_twitch, cy - 25)
        inner_l.lineTo(cx - 8, cy - 18)
        inner_l.closeSubpath()
        painter.drawPath(inner_l)

        inner_r = QPainterPath()
        inner_r.moveTo(cx + 8, cy - 18)
        inner_r.lineTo(cx + 22 - ear_twitch, cy - 25)
        inner_r.lineTo(cx + 16, cy - 12)
        inner_r.closeSubpath()
        painter.drawPath(inner_r)

        # Eyes
        eye_y = cy - 2
        painter.setPen(QPen(QColor(40, 30, 20), 2.0))
        if self.blink_state == 0 and self.state != "sleeping":
            painter.setBrush(QBrush(QColor(40, 30, 20)))
            painter.drawEllipse(QPoint(int(cx - 9), int(eye_y)), 3, 5)
            painter.drawEllipse(QPoint(int(cx + 9), int(eye_y)), 3, 5)
            # Specular
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(int(cx - 10), int(eye_y - 2)), 1, 1)
            painter.drawEllipse(QPoint(int(cx + 8), int(eye_y - 2)), 1, 1)
        else:
            # Crescent happy eyes
            painter.drawLine(int(cx - 12), int(eye_y), int(cx - 6), int(eye_y - 2))
            painter.drawLine(int(cx - 6), int(eye_y - 2), int(cx - 4), int(eye_y))
            painter.drawLine(int(cx + 4), int(eye_y), int(cx + 6), int(eye_y - 2))
            painter.drawLine(int(cx + 6), int(eye_y - 2), int(cx + 12), int(eye_y))

        # Nose & Whiskers
        painter.setPen(QPen(QColor(255, 120, 140), 1.5))
        painter.drawPoint(int(cx), int(cy + 4))

        painter.setPen(QPen(QColor(180, 170, 160), 1.0))
        painter.drawLine(int(cx - 14), int(cy + 4), int(cx - 24), int(cy + 2))
        painter.drawLine(int(cx - 14), int(cy + 7), int(cx - 24), int(cy + 9))
        painter.drawLine(int(cx + 14), int(cy + 4), int(cx + 24), int(cy + 2))
        painter.drawLine(int(cx + 14), int(cy + 7), int(cx + 24), int(cy + 9))

    def _draw_hacker(self, painter: QPainter, cx: float, cy: float):
        """Draw cyberpunk hacker sprite with neon visor and headset."""
        # Helmet / Head
        painter.setPen(QPen(QColor(20, 255, 200), 1.5))
        painter.setBrush(QBrush(QColor(25, 28, 38)))
        painter.drawRoundedRect(QRectF(cx - 22, cy - 20, 44, 40), 10, 10)

        # Neon Cyan Visor
        gradient = QLinearGradient(cx - 18, cy - 8, cx + 18, cy + 4)
        gradient.setColorAt(0.0, QColor(0, 255, 220, 220))
        gradient.setColorAt(1.0, QColor(0, 160, 255, 220))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cx - 18, cy - 8, 36, 12), 4, 4)

        # Glowing waveform on visor
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        wave_y = cy - 2 + math.sin(self.anim_tick * 0.2) * 2.0
        painter.drawLine(int(cx - 12), int(cy - 2), int(cx - 4), int(wave_y))
        painter.drawLine(int(cx - 4), int(wave_y), int(cx + 4), int(cy - 2))
        painter.drawLine(int(cx + 4), int(cy - 2), int(cx + 12), int(wave_y))

        # Headset mic
        painter.setPen(QPen(QColor(255, 100, 50), 2.0))
        painter.drawLine(int(cx - 22), int(cy + 2), int(cx - 22), int(cy + 14))
        painter.drawLine(int(cx - 22), int(cy + 14), int(cx - 10), int(cy + 16))
        painter.setBrush(QBrush(QColor(255, 100, 50)))
        painter.drawEllipse(QPoint(int(cx - 8), int(cy + 16)), 3, 3)

    def _draw_zen(self, painter: QPainter, cx: float, cy: float):
        """Draw serene floating cloud / meditating monk holding hot tea."""
        # Cloud body
        painter.setPen(QPen(QColor(180, 210, 220), 1.5))
        gradient = QLinearGradient(cx, cy - 20, cx, cy + 20)
        gradient.setColorAt(0.0, QColor(255, 255, 255, 240))
        gradient.setColorAt(1.0, QColor(230, 245, 250, 240))
        painter.setBrush(QBrush(gradient))

        path = QPainterPath()
        path.addEllipse(QPoint(int(cx), int(cy)), 22, 18)
        path.addEllipse(QPoint(int(cx - 14), int(cy + 2)), 14, 12)
        path.addEllipse(QPoint(int(cx + 14), int(cy + 2)), 14, 12)
        painter.drawPath(path)

        # Closed serene meditating eyes
        painter.setPen(QPen(QColor(80, 110, 120), 1.8))
        painter.drawArc(int(cx - 14), int(cy - 6), 8, 6, 0, -180 * 16)
        painter.drawArc(int(cx + 6), int(cy - 6), 8, 6, 0, -180 * 16)

        # Gentle smile
        painter.drawArc(int(cx - 4), int(cy + 2), 8, 5, 0, -180 * 16)

        # Steaming tea cup
        cup_x = cx + 18
        cup_y = cy + 12
        painter.setPen(QPen(QColor(120, 150, 160), 1.2))
        painter.setBrush(QBrush(QColor(240, 230, 210)))
        painter.drawRoundedRect(QRectF(cup_x - 5, cup_y - 4, 10, 8), 2, 2)

        # Steam line
        steam_offset = math.sin(self.anim_tick * 0.15) * 2.0
        painter.setPen(QPen(QColor(180, 200, 210, 160), 1.2))
        painter.drawLine(int(cup_x), int(cup_y - 6), int(cup_x + steam_offset), int(cup_y - 12))

    # -------------------------------------------------------------
    # Mouse Interaction & Context Menus
    # -------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            new_pos = event.globalPosition().toPoint() - self.drag_position
            self.move(new_pos)
            self._update_bubble_position()
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        self._save_config()

    def mouseDoubleClickEvent(self, event):
        """Double click: Trigger a deep thought based on current app."""
        thought = self.brain.think(self.last_context)
        self.say(thought, duration_sec=8.0)

    def contextMenuEvent(self, event):
        """Right click context menu."""
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #ffffff;
                border: 1px solid #d4d4cc;
                border-radius: 8px;
                padding: 4px;
                font-size: 13px;
                color: #222222;
            }
            QMenu::item {
                padding: 6px 18px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #f0f0ea;
                color: #ff6600;
            }
            """
        )

        # Personality submenu
        person_menu = menu.addMenu("切换角色 (Personality)")
        for key, pinfo in PERSONALITIES.items():
            action = person_menu.addAction(f"{pinfo['avatar_symbol']} {pinfo['name']}")
            action.setCheckable(True)
            action.setChecked(self.brain.personality == key)
            action.triggered.connect(lambda checked, k=key: self._switch_personality(k))

        menu.addSeparator()

        # Action: Coffee feed
        coffee_action = menu.addAction("投喂热咖啡 (Focus Boost)")
        coffee_action.triggered.connect(self._feed_coffee)

        # Action: Toggle sleep
        sleep_label = "唤醒精灵 (Wake Up)" if self.state == "sleeping" else "让它小憩 (Take a Nap)"
        sleep_action = menu.addAction(sleep_label)
        sleep_action.triggered.connect(self._toggle_sleep)

        menu.addSeparator()

        # Action: Close bubble
        close_bubble_action = menu.addAction("关闭气泡 (Hide Bubble)")
        close_bubble_action.triggered.connect(self.bubble.hide_bubble)

        # Action: Quit
        quit_action = menu.addAction("退出精灵 (Quit)")
        quit_action.triggered.connect(self._quit_app)

        menu.exec(QCursor.pos())

    def _switch_personality(self, key: str):
        self.brain.set_personality(key)
        self._save_config()
        self.update()
        pinfo = self.brain.get_personality_info()
        self.say(f"已切换为：{pinfo['name']}！{pinfo['description']}")

    def _feed_coffee(self):
        self.state = "idle"
        replies = {
            "clippy": "咕噜咕噜~ 咖啡因注入完毕！感觉现在能一口气排查十个 Bug！",
            "cat": "喵呜！虽然猫猫不能喝咖啡，但本喵闻着挺香的，奖励你揉肚子！",
            "hacker": "双份浓缩冰美式已就绪。键盘敲击速度提升 200%！",
            "zen": "茶禅一味，暖意自心底升起。愿你保持内心的从容。",
        }
        self.say(replies.get(self.brain.personality, "能量满满！继续加油！"))

    def _toggle_sleep(self):
        if self.state == "sleeping":
            self.state = "idle"
            self.say("（揉了揉眼睛）伸个懒腰，我醒啦！")
        else:
            self.state = "sleeping"
            self.say("呼……好困哦，本精灵先眯一会儿，有事双击叫醒我~")

    def _quit_app(self):
        self.sensor.stop()
        self._save_config()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Desktop Pet")

    pet = DesktopPet()
    pet.show()
    pet.raise_()
    pet.activateWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
