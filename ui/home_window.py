from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .training_window import TrainingWindow
from ui.coach_window import CoachWindow


class HomeWindow(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        
        
        self.setWindowTitle("سیستم تمرین هوشمند 🏋️")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        header = QLabel("به سیستم تمرین هوشمند خوش آمدید")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 32px;
                font-weight: bold;
                margin: 20px;
            }
        """)
        main_layout.addWidget(header)
        
        description = QLabel("این سیستم به شما کمک می‌کند تا تمرینات خود را با دقت و اصولی انجام دهید")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 18px;
                margin: 10px;
            }
        """)
        main_layout.addWidget(description)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 15px;
                padding: 30px;
                font-size: 20px;
                min-width: 200px;
                min-height: 150px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
        """
        
        self.training_btn = QPushButton("تمرین کردن ")
        self.training_btn.setStyleSheet(button_style)
        self.training_btn.clicked.connect(self.open_training)
        button_layout.addWidget(self.training_btn)
        
        self.stickman_btn = QPushButton("استیکمن ")
        self.stickman_btn.setStyleSheet(button_style)
        button_layout.addWidget(self.stickman_btn)
        
        self.coach_btn = QPushButton("ارتباط با مربی ")
        self.coach_btn.setStyleSheet(button_style)
        self.coach_btn.clicked.connect(self.open_coach)
        button_layout.addWidget(self.coach_btn)
        
        main_layout.addLayout(button_layout)
        
        footer = QLabel("طراحی و توسعه: حسین قربانی")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("""
            QLabel {
                color: #95a5a6;
                font-size: 14px;
                margin: 20px;
            }
        """)
        main_layout.addWidget(footer)
        
        font = QFont()
        font.setFamily("B Nazanin")
        font.setPointSize(14)
        self.setFont(font)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
        """)
        
    def open_training(self):
        
        self.training_window = TrainingWindow(self.detector, parent=self)
        self.training_window.show()
        self.hide()

    def open_coach(self):
        
        self.coach_window = CoachWindow(parent=self)
        self.coach_window.show()
        self.hide() 