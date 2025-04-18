from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                          QPushButton, QLabel, QComboBox,QFileDialog, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2

class TrainingWindow(QMainWindow):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        
        self.home_window = parent
        
        self.setWindowTitle("تمرین")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.exercise_type_frame = QFrame()
        self.exercise_type_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.exercise_type_frame.setLineWidth(2)
        self.exercise_type_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.7);
                border: 2px solid #2ecc71;
                border-radius: 10px;
            }
        """)
        
        exercise_type_layout = QVBoxLayout(self.exercise_type_frame)
        self.exercise_type_label = QLabel("حرکت: نامشخص")
        self.exercise_type_label.setAlignment(Qt.AlignCenter)
        self.exercise_type_label.setStyleSheet("""
            QLabel {
                font-family: 'B Nazanin';
                font-size: 20px;
                color: white;
                padding: 10px;
            }
        """)
        exercise_type_layout.addWidget(self.exercise_type_label)
        
        self.exercise_type_frame.setFixedSize(200, 60)
        
        self.exercise_type_frame.setGeometry(300, 20, 200, 60)
        
        self.rep_label = QLabel("", self)
        self.rep_label.setAlignment(Qt.AlignCenter)
        self.rep_label.setStyleSheet("""
            QLabel {
                font-family: 'B Nazanin';
                font-size: 24px;
                color: #2ecc71;
                background-color: rgba(245, 245, 245, 0.7);
                padding: 10px;
                border-radius: 10px;
            }
        """)
        
        self.rep_label.setGeometry(300, 50, 200, 50)
        
        self.showFullScreen()
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label, stretch=1)
        
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(20)
        
        button_style = """
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 18px;
                min-width: 150px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """
        
        self.auto_detect_btn = QPushButton("تشخیص خودکار 🔄")
        self.auto_detect_btn.setStyleSheet(button_style)
        self.auto_detect_btn.setCheckable(True)
        self.auto_detect_btn.clicked.connect(self.toggle_auto_detect)
        control_layout.addWidget(self.auto_detect_btn)
        
        self.exercise_combo = QComboBox()
        self.exercise_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
                min-width: 200px;
                min-height: 50px;
                font-size: 16px;
            }
        """)
        self.exercise_combo.addItems([
            "تشخیص خودکار 🎯",
            "پوش‌آپ 💪",
            "پول‌آپ 🏋️",
            "ددلیفت 🏋️‍♂️",
            "اسکات 🦵",
            
        ])
        control_layout.addWidget(self.exercise_combo)
        
        self.start_button = QPushButton("شروع 🎬")
        self.start_button.setStyleSheet(button_style)
        self.stop_button = QPushButton("توقف ⏹")
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.setEnabled(False)
        
        self.back_button = QPushButton("برگشت 🔙")
        self.back_button.setStyleSheet(button_style)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.back_button)
        
        self.video_button = QPushButton("انتخاب ویدیو 📁")
        self.video_button.setStyleSheet(button_style)
        self.video_button.clicked.connect(self.select_video)
        control_layout.addWidget(self.video_button)
        
        self.exercise_names = {
     
        }
        
        counter_layout = QHBoxLayout()
        
        control_layout.addLayout(counter_layout)
        
        layout.addWidget(control_panel)
        
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        
        self.start_button.clicked.connect(self.start_exercise)
        self.stop_button.clicked.connect(self.stop_exercise)
        self.back_button.clicked.connect(self.close_window)
        
        self.cap = None
        self.auto_detect = False
        self.video_path = None
        
        font = QFont()
        font.setFamily("B Nazanin")
        font.setPointSize(12)
        self.setFont(font)
        
    def toggle_auto_detect(self):
      
        self.auto_detect = self.auto_detect_btn.isChecked()
        self.exercise_combo.setEnabled(not self.auto_detect)
        
    def select_video(self):
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "انتخاب ویدیو",
            "",
            "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.start_button.setEnabled(True)
        
    def start_exercise(self):
        
        if self.video_path:
           
            self.cap = cv2.VideoCapture(self.video_path)
        else:
           
            self.cap = cv2.VideoCapture(0)
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30 
        
        self.camera_timer.start(int(1000/self.fps))  
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video_button.setEnabled(False)
        
    def stop_exercise(self):
        
        self.camera_timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_button.setEnabled(True)
        
    def update_frame(self):
        
        ret, frame = self.cap.read()
        if ret:
          
            processed_frame, exercise_type, rep_count = self.detector.process_frame(frame, True)
            
            if exercise_type:
                persian_name = self.exercise_names.get(exercise_type, 'نامشخص')
                self.exercise_type_label.setText(f"حرکت: {persian_name}")
               
            
            if rep_count is not None: 
                self.rep_label.setText(f"تعداد حرکات: {rep_count}")
            
            
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)
        else:
            
            if self.video_path:
                self.stop_exercise()
                self.video_button.setEnabled(True)
            
    def close_window(self):
        
        self.stop_exercise()
        if self.home_window:
            self.home_window.show()  
        self.close()
        
    def closeEvent(self, event):
      
        self.stop_exercise()
        event.accept() 