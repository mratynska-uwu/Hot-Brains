"""Valence-arousal plane: target from the image ratings vs. the state decoded from EEG."""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

TARGET_COLOR = "#ffb86b"
DECODED_COLOR = "#6dd7bf"


class AffectPlane(QWidget):
    """x = valence (negative -> positive), y = arousal (low -> high), both in [-1, 1].

    The target marker is the image's rated valence/arousal mapped so that 0 is the
    class boundary. The decoded marker is 2 * P(class 1) - 1 from the classifiers,
    i.e. their confidence, not a calibrated rating.
    """

    def __init__(self, x_name: str = "valence", y_name: str = "arousal"):
        super().__init__()
        self.x_name, self.y_name = x_name, y_name
        self.target: tuple[float, float] | None = None
        self.decoded: tuple[float, float] | None = None
        self.trail: deque[tuple[float, float]] = deque(maxlen=40)
        self.caption = ""
        self.on_target_distance = 0.35
        self.setMinimumSize(300, 330)

    def set_axes(self, x_name: str, y_name: str):
        self.x_name, self.y_name = x_name, y_name
        self.update()

    def set_target(self, point: tuple[float, float] | None):
        self.target = point
        self.update()

    def set_decoded(self, point: tuple[float | None, float | None] | None, keep_trail: bool = True):
        if point is None or point[0] is None or point[1] is None:
            self.decoded = None
        else:
            self.decoded = (float(point[0]), float(point[1]))
            if keep_trail:
                self.trail.append(self.decoded)
        self.update()

    def distance(self) -> float | None:
        if self.target is None or self.decoded is None:
            return None
        return float(((self.target[0] - self.decoded[0]) ** 2 + (self.target[1] - self.decoded[1]) ** 2) ** .5)

    def on_target(self) -> bool:
        distance = self.distance()
        return distance is not None and distance < self.on_target_distance

    def decoded_color(self) -> str:
        """Green on target, yellow when close, red when far; neutral without a target."""
        distance = self.distance()
        if distance is None:
            return DECODED_COLOR
        if distance < self.on_target_distance:
            return "#3ddc84"
        return "#f4e06d" if distance < 2 * self.on_target_distance else "#ff727a"

    def set_caption(self, text: str):
        self.caption = text
        self.update()

    def clear(self):
        self.target = None
        self.decoded = None
        self.trail.clear()
        self.caption = ""
        self.update()

    def _square(self) -> QRectF:
        side = min(self.width() - 60, self.height() - 100)
        return QRectF((self.width() - side) / 2, 30, side, side)

    def _to_px(self, square: QRectF, point: tuple[float, float]) -> QPointF:
        x = max(-1.0, min(1.0, point[0]))
        y = max(-1.0, min(1.0, point[1]))
        return QPointF(square.center().x() + x * square.width() / 2, square.center().y() - y * square.height() / 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101827"))
        square = self._square()
        painter.setPen(QPen(QColor("#627795"), 2))
        painter.setBrush(QColor("#142638"))
        painter.drawRect(square)
        painter.setPen(QPen(QColor("#3d4f69"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(square.center().x(), square.top()), QPointF(square.center().x(), square.bottom()))
        painter.drawLine(QPointF(square.left(), square.center().y()), QPointF(square.right(), square.center().y()))

        painter.setFont(QFont("Arial", 10))
        painter.setPen(QColor("#a3acbd"))
        painter.drawText(QRectF(square.left(), square.bottom() + 2, square.width(), 18),
                         Qt.AlignmentFlag.AlignCenter, f"{self.x_name}  →")
        painter.drawText(QRectF(square.left(), square.bottom() + 2, square.width(), 18),
                         Qt.AlignmentFlag.AlignLeft, "negative")
        painter.drawText(QRectF(square.left(), square.bottom() + 2, square.width(), 18),
                         Qt.AlignmentFlag.AlignRight, "positive")
        painter.drawText(QRectF(square.left(), 8, square.width(), 18), Qt.AlignmentFlag.AlignCenter,
                         f"↑  {self.y_name} (high)")
        painter.drawText(QRectF(square.left(), square.bottom() - 18, square.width(), 16),
                         Qt.AlignmentFlag.AlignCenter, "low")

        if len(self.trail) > 1:
            points = list(self.trail)
            for k in range(1, len(points)):
                color = QColor(DECODED_COLOR)
                color.setAlpha(int(25 + 150 * k / len(points)))
                painter.setPen(QPen(color, 2))
                painter.drawLine(self._to_px(square, points[k - 1]), self._to_px(square, points[k]))
        if self.target is not None:
            center = self._to_px(square, self.target)
            painter.setPen(QPen(QColor(TARGET_COLOR), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 18, 18)
            painter.drawLine(center + QPointF(-8, 0), center + QPointF(8, 0))
            painter.drawLine(center + QPointF(0, -8), center + QPointF(0, 8))
        if self.decoded is not None:
            painter.setPen(QPen(QColor("#0b1a2a"), 3))
            painter.setBrush(QColor(self.decoded_color()))
            painter.drawEllipse(self._to_px(square, self.decoded), 16, 16)

        painter.setFont(QFont("Arial", 11, QFont.Weight.DemiBold))
        y = square.bottom() + 24
        painter.setPen(QColor(TARGET_COLOR))
        target_text = "target —" if self.target is None else f"◯ target  v {self.target[0]:+.2f}  a {self.target[1]:+.2f}"
        painter.drawText(QRectF(0, y, self.width() / 2, 20), Qt.AlignmentFlag.AlignCenter, target_text)
        painter.setPen(QColor(DECODED_COLOR))
        decoded_text = "EEG —" if self.decoded is None else f"● EEG  v {self.decoded[0]:+.2f}  a {self.decoded[1]:+.2f}"
        painter.drawText(QRectF(self.width() / 2, y, self.width() / 2, 20), Qt.AlignmentFlag.AlignCenter, decoded_text)
        if self.caption:
            painter.setPen(QColor("#e8effc"))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(QRectF(0, y + 20, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self.caption)
        painter.end()
