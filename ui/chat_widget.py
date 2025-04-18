from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, 
                           QTextEdit, QPushButton, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import datetime

class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f6fa;
            }
            QScrollBar:vertical {
                border: none;
                background: #f5f6fa;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #b2bec3;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.addStretch()
        
        self.chat_content.setStyleSheet("""
            QWidget {
                background-color: #f5f6fa;
            }
        """)
        
        self.chat_area.setWidget(self.chat_content)
        main_layout.addWidget(self.chat_area)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #dcdde1;")
        main_layout.addWidget(separator)
        
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        
        self.voice_btn = QPushButton()
        self.voice_btn.setIcon(QIcon("icons/mic.png"))
        self.voice_btn.setFixedSize(40, 40)
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
        """)
        input_layout.addWidget(self.voice_btn)
        
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("پیام خود را بنویسید...")
        self.message_input.setMaximumHeight(80)
        self.message_input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #dcdde1;
                border-radius: 15px;
                padding: 10px;
                font-size: 14px;
                background-color: white;
            }
            QTextEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        input_layout.addWidget(self.message_input)
        
        self.send_btn = QPushButton("ارسال")
        self.send_btn.setFixedSize(60, 40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
        """)
        input_layout.addWidget(self.send_btn)
        
        main_layout.addLayout(input_layout)
        
    def add_message(self, text, is_user=True):
   
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        
        message_container = QWidget()
        message_layout = QVBoxLayout(message_container)
        message_layout.setContentsMargins(5, 5, 5, 5)
        
        message = QLabel(text)
        message.setWordWrap(True)
        message.setStyleSheet(f"""
            QLabel {{
                background-color: {'#3498db' if is_user else '#ffffff'};
                color: {'white' if is_user else 'black'};
                border-radius: 15px;
                padding: 10px;
                font-size: 14px;
                max-width: 70%;
            }}
        """)
        
        time = QLabel(datetime.datetime.now().strftime("%H:%M"))
        time.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 11px;
                padding: 2px;
            }
        """)
        
        if is_user:
            message.setAlignment(Qt.AlignRight)
            time.setAlignment(Qt.AlignRight)
            message_layout.addWidget(message)
            message_layout.addWidget(time)
        else:
            message.setAlignment(Qt.AlignLeft)
            time.setAlignment(Qt.AlignLeft)
            message_layout.addWidget(message)
            message_layout.addWidget(time)
        
        self.chat_layout.addWidget(message_container)
        self.chat_layout.addStretch()
        
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
        
    def add_user_message(self, text):
   
        self.add_message(text, True)
        
    def add_coach_message(self, text):
       
        self.add_message(text, False)
        
    def add_status_message(self, text):
   
        message = QLabel(text)
        message.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-style: italic;
                padding: 5px;
                background-color: #f5f6fa;
                border-radius: 10px;
            }
        """)
        message.setAlignment(Qt.AlignCenter)
        
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        self.chat_layout.addWidget(message)
        self.chat_layout.addStretch()
        
    def add_error_message(self, text):
       
        message = QLabel(text)
        message.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
                padding: 5px;
                background-color: #fde8e8;
                border-radius: 10px;
            }
        """)
        message.setAlignment(Qt.AlignCenter)
        
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        self.chat_layout.addWidget(message)
        self.chat_layout.addStretch()
