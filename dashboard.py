"""Desktop interface for the webcam tracker."""
import base64
import time
import tkinter as tk
from collections import deque

import cv2


BACKGROUND = "#101522"
PANEL = "#1a2233"
TEXT = "#f3f6fd"
MUTED = "#a9b7cf"
ACCENT = "#83e3c0"
METRICS = (
    ("Моргания", "в минуту"),
    ("Скорость поворота головы", "°/с"),
    ("Открытие рта", "Δ к базовому уровню"),
    ("Положение бровей", "Δ к базовому уровню"),
)
STATUSES = {
    "Tracking": "Лицо отслеживается",
    "Show exactly one face": "В кадре должно быть одно лицо",
    "Move closer and center your face": "Подвиньтесь ближе и сядьте по центру",
    "Head pose unavailable": "Положение головы не определено",
    "Face the camera": "Посмотрите прямо в камеру",
}


class Dashboard:
    def __init__(self, root, camera, tracker, args, writer):
        self.root = root
        self.camera = camera
        self.tracker = tracker
        self.args = args
        self.writer = writer
        self.history = deque()
        self.last_sample = float("-inf")
        self.started = None
        self.ready = False
        self.image = None

        root.title("Face Motion Lab")
        root.geometry("1080x680")
        root.minsize(780, 590)
        root.configure(bg=BACKGROUND)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        header = tk.Frame(root, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 14))
        self.label(header, "Анализ движений лица", 22, bg=BACKGROUND, bold=True).pack(anchor="w")
        self.status = self.label(header, "", 12, bg=BACKGROUND, fg=ACCENT)
        self.status.pack(anchor="w", pady=(5, 0))

        content = tk.Frame(root, bg=BACKGROUND)
        content.grid(row=1, column=0, sticky="nsew", padx=24)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self.video = tk.Canvas(content, bg="#080b12", highlightthickness=0)
        self.video.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.video_item = self.video.create_image(0, 0)

        sidebar = tk.Frame(content, bg=PANEL, padx=16, pady=12)
        sidebar.grid(row=0, column=1, sticky="ns")
        self.readings = []
        for title, unit in METRICS:
            self.label(sidebar, title, 11, fg=MUTED).pack(anchor="w", pady=(3, 0))
            reading = self.label(sidebar, "—", 19, bold=True)
            reading.pack(anchor="w")
            self.label(sidebar, unit, 9, fg=MUTED).pack(anchor="w", pady=(0, 7))
            self.readings.append(reading)
        self.label(sidebar, "График частоты · 10 с", 10, fg=MUTED).pack(anchor="w")
        self.graph = tk.Canvas(sidebar, width=232, height=90, bg=BACKGROUND, highlightthickness=0)
        self.graph.pack(fill="x", pady=(6, 0))

        footer = tk.Frame(root, bg=BACKGROUND)
        footer.grid(row=2, column=0, sticky="ew", padx=24, pady=16)
        self.action = self.button(
            footer, "Начать запись" if writer else "Калибровать заново",
            self.begin_trial if writer else self.recalibrate,
        )
        self.action.pack(side="left")
        if writer:
            self.action.configure(state="disabled")
        self.button(footer, "Закрыть", root.destroy, primary=False).pack(side="right")
        root.bind("<Escape>", lambda _: root.destroy())
        root.bind("<q>", lambda _: root.destroy())
        root.bind("<space>", self.on_space)
        root.bind("<r>", lambda _: self.recalibrate())
        root.after_idle(self.tick)

    @staticmethod
    def label(parent, text, size, bg=PANEL, fg=TEXT, bold=False):
        return tk.Label(parent, text=text, bg=bg, fg=fg,
                        font=("Segoe UI", size, "bold" if bold else "normal"))

    @staticmethod
    def button(parent, text, command, primary=True):
        return tk.Button(parent, text=text, command=command,
                         bg=ACCENT if primary else PANEL, fg=BACKGROUND if primary else TEXT,
                         activebackground="#b2f4dd", activeforeground=BACKGROUND,
                         disabledforeground="#485969", relief="flat", padx=16, pady=7,
                         font=("Segoe UI", 11, "bold"), cursor="hand2")

    def on_space(self, event):
        if not isinstance(event.widget, tk.Button):
            self.begin_trial()

    def recalibrate(self):
        if self.writer is None:
            self.tracker.reset()
            self.history.clear()

    def begin_trial(self):
        if self.writer is not None and self.ready and self.started is None:
            self.started = time.monotonic()
            self.action.configure(text="Завершить запись", command=self.root.destroy)

    def tick(self):
        ok, frame = self.camera.read()
        if not ok:
            self.status.configure(text="Камера перестала передавать кадры")
            self.action.configure(state="disabled")
            self.video.itemconfigure(self.video_item, image="")
            for reading in self.readings:
                reading.configure(text="—")
            self.graph.delete("all")
            return

        now = time.monotonic()
        status, values, bbox = self.tracker.measure(frame, now)
        self.ready = values is not None and all(value is not None for value in values)
        message = STATUSES.get(status, status)
        if status.startswith("Look forward"):
            message = "Калибровка · расслабьте лицо и смотрите прямо " + status[status.index("("):]
        elif status == "Tracking" and not self.ready:
            message = "Лицо отслеживается · подождите, пока появится частота моргания"
        if self.started is not None:
            message = f"Запись {self.args.trial} · {now - self.started:.0f} с  |  {message}"
        self.status.configure(text=message, fg=ACCENT if self.ready else MUTED)
        if self.writer and self.started is None:
            self.action.configure(state="normal" if self.ready else "disabled")
        for index, reading in enumerate(self.readings):
            value = values[index] if values is not None else None
            reading.configure(text="—" if value is None else
                              f"{value:.1f}" if index < 2 else f"{value:+.3f}")

        if now - self.last_sample >= 0.1:
            self.history.append((now, values[0] if values is not None else None))
            while self.history and now - self.history[0][0] > 10:
                self.history.popleft()
            if self.started is not None and self.ready:
                self.writer.writerow((self.args.participant, self.args.trial, self.args.label,
                                      now - self.started, *values))
            self.last_sample = now
            self.draw_graph(now)
        if bbox:
            cv2.rectangle(frame, *bbox, (192, 227, 131), 2)
        self.show_frame(frame)
        self.root.after(15, self.tick)

    def show_frame(self, frame):
        width = max(1, self.video.winfo_width())
        height = max(1, self.video.winfo_height())
        scale = min(width / frame.shape[1], height / frame.shape[0])
        dimensions = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
        preview = cv2.resize(frame, dimensions, interpolation=cv2.INTER_LINEAR)
        ok, png = cv2.imencode(".png", preview, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if ok:
            self.image = tk.PhotoImage(data=base64.b64encode(png.tobytes()), format="png")
            self.video.coords(self.video_item, width / 2, height / 2)
            self.video.itemconfigure(self.video_item, image=self.image)

    def draw_graph(self, now):
        self.graph.delete("all")
        width = max(1, self.graph.winfo_width())
        rates = [rate for _, rate in self.history if rate is not None]
        ceiling = max(30, max(rates, default=0))
        for y in (20, 45, 70):
            self.graph.create_line(30, y, width - 6, y, fill="#28354d")
        self.graph.create_text(25, 20, text=f"{ceiling:.0f}", anchor="e", fill=MUTED)
        self.graph.create_text(25, 70, text="0", anchor="e", fill=MUTED)
        previous = None
        for when, rate in self.history:
            if rate is None:
                previous = None
                continue
            point = (30 + (width - 36) * (1 - (now - when) / 10), 70 - 50 * rate / ceiling)
            if previous is not None:
                self.graph.create_line(*previous, *point, fill=ACCENT, width=2)
            previous = point
