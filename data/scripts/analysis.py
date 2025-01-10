import sys
import threading
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QTextEdit
from PySide6.QtCore import Signal, Slot, Qt
from PySide6 import QtGui, QtCore
import cv2
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from numpy import linspace

class VideoProcessor(QWidget):
    update_label_signal = Signal(str)
    update_output_signal = Signal(str)
    update_frame_signal = Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.is_running = False
        self.video_cap = None
        self.writer = None
        self.tracker = None
        self.model = None
        self.file_name = ""
        self.output_file_name = ""
        self.program_start = None
        self.last_frame_time = None
        self.current_frame = 0
        self.frame_count = 0

    def initUI(self):
        self.setWindowTitle('Football Player Detection')
        self.setMinimumSize(1280, 820)
        self.setGeometry(100, 100, 1280, 820)

        self.model_path = ""

        self.txt_edit = QTextEdit(self)
        self.txt_edit.setReadOnly(True)
        self.txt_edit.setMinimumHeight(100)

        sys.stdout = EmittingStream()
        sys.stdout.text_written.connect(self.append_output)

        self.choose_file_btn = QPushButton('Choose File', self)
        self.choose_file_btn.setFixedWidth(150)
        self.choose_file_btn.clicked.connect(self.choose_file)

        self.save_file_btn = QPushButton('Save File', self)
        self.save_file_btn.setFixedWidth(150)
        self.save_file_btn.clicked.connect(self.choose_output_file)
        self.save_file_btn.setEnabled(False)

        self.start_btn = QPushButton('Start', self)
        self.start_btn.setFixedWidth(150)
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton('Stop', self)
        self.stop_btn.setFixedWidth(150)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        self.status_label = QLabel('No file selected', self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 650)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_file_btn)
        button_layout.addWidget(self.save_file_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.video_label, 2)
        layout.addWidget(self.txt_edit, 1)

        layout.setContentsMargins(0, 10, 0, 0)

        self.setLayout(layout)

        self.update_label_signal.connect(self.update_status)
        self.update_output_signal.connect(self.append_output)
        self.update_frame_signal.connect(self.update_frame)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        sys.stdout = sys.__stdout__
        event.accept()

    @Slot(str)
    def append_output(self, text):
        cursor = self.txt_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.txt_edit.setTextCursor(cursor)
        self.txt_edit.ensureCursorVisible()

    @Slot(QtGui.QImage)
    def update_frame(self, frame):
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(frame))

    @Slot()
    def choose_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Choose Video File', '', 'Videos (*.mp4 *.avi)')
        if self.file_name:
            self.status_label.setText(f'File selected: {self.file_name}')
            self.start_btn.setEnabled(True)

    @Slot()
    def choose_output_file(self):
        self.output_file_name, _ = QFileDialog.getSaveFileName(self, 'Save Output File', '', 'MP4 Files (*.mp4)')
        if self.output_file_name:
            if not self.output_file_name.endswith('.mp4'):
                self.output_file_name += '.mp4'
            self.status_label.setText(f'Output file selected: {self.output_file_name}')

    @Slot()
    def start_processing(self):
        if self.file_name:
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.save_file_btn.setEnabled(False)
            self.status_label.setText("Processing started...")

            threading.Thread(target=self.process_video, daemon=True).start()

    @Slot()
    def stop_processing(self):
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Processing paused.")
        self.enable_save_button()

    @Slot()
    def update_status(self, text):
        self.status_label.setText(text)

    def process_video(self):
        PITCH_LENGTH = 105
        PITCH_WIDTH = 60
        FIRST_GOAL_COORDS = [890, 255, 1070, 315, 5]
        GOAL_1_HEIGHT = 135
        CONFIDENCE_THRESHOLD = 0.75
        MAX_AGE = 50
        DARK_BLUE = (27, 45, 166)
        WHITE = (255, 255, 255)
        x_values = linspace(FIRST_GOAL_COORDS[0], FIRST_GOAL_COORDS[2], num=360)
        slope = (FIRST_GOAL_COORDS[3] - FIRST_GOAL_COORDS[1]) / (FIRST_GOAL_COORDS[2] - FIRST_GOAL_COORDS[0])
        y_values = slope * (x_values - FIRST_GOAL_COORDS[0]) + FIRST_GOAL_COORDS[1]
        self.current_frame = 0
        print_objects = []

        self.video_cap = cv2.VideoCapture(self.file_name)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = create_video_writer(self.video_cap, self.output_file_name)

        self.model = YOLO(self.model_path)
        self.tracker = DeepSort(MAX_AGE)

        self.last_frame_time = datetime.datetime.now()

        while self.is_running:
            self.current_frame += 1
            ret, frame = self.video_cap.read()
            if not ret:
                break

            detections = self.model(frame)[0]
            results = []

            for data in detections.boxes.data.tolist():
                confidence = data[4]
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = "Player" if int(data[5]) == 1 else "Ball" if int(data[5]) == 0 else str(int(data[5]))
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

            tracks = self.tracker.update_tracks(results, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                print_objects.append(f"{track_id} {class_id} [{int(ltrb[0])}, {int(ltrb[1])}, {int(ltrb[2])}, {int(ltrb[3])}]")

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), DARK_BLUE, 2)

            current_time = datetime.datetime.now()
            fps = round(self.calculate_fps(current_time), 2)
            self.last_frame_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.update_frame_signal.emit(qt_image)

            writer.write(frame)
            print(f"FPS: {fps}, Frame: {self.current_frame}/{self.frame_count}")
            print(print_objects)
            print("")
            print_objects = []

            if self.current_frame / self.frame_count >= 0.25:
                self.enable_save_button()

        self.video_cap.release()
        writer.release()
        cv2.destroyAllWindows()
        self.enable_save_button()

    def enable_save_button(self):
        if (self.current_frame / self.frame_count) >= 0.25 and not self.is_running:
            self.save_file_btn.setEnabled(True)
        else:
            self.save_file_btn.setEnabled(False)

    def calculate_fps(self, current_time):
        if self.last_frame_time is None:
            return 0.0
        time_diff = (current_time - self.last_frame_time).total_seconds()
        if time_diff > 0:
            return 1.0 / time_diff
        return 0.0

class EmittingStream(QtCore.QObject):
    text_written = QtCore.Signal(str)

    def write(self, text):
        self.text_written.emit(str(text))

def create_video_writer(video_cap, output_filename):

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

def Analysis():
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec())