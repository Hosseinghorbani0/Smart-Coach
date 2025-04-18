from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QScrollArea, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
from core.coach_manager import CoachManager
import os
import wave
import pyaudio
from core.config import Config
import tempfile
from openai import OpenAI
import time
import numpy as np

class AnimatedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
    def fade_in(self):
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.start()
        
    def fade_out(self):
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(1)
        self.anim.setEndValue(0)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.start()

class CoachWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("مربی هوشمند ")
        self.setMinimumSize(800, 600)
        
        self.coach_manager = CoachManager()
        
        self.font = QFont()
        self.font.setFamily("B Nazanin")
        self.font.setPointSize(14)
        self.setFont(self.font)
        
        self.is_recording = False
        
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.CHUNK = 4096  
        self.FORMAT = pyaudio.paFloat32 
        self.CHANNELS = 1 
        self.RATE = 16000  
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        self.init_ui()
        
    def init_ui(self):
       
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #ccc;
                border-radius: 15px;
                background-color: white;
            }
        """)
        
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.chat_area.setWidget(self.chat_widget)
        layout.addWidget(self.chat_area)
        
        bottom_layout = QHBoxLayout()
        
        self.voice_btn = QPushButton("⚫")
        self.voice_btn.setFixedSize(50, 50)
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 5px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.voice_btn.clicked.connect(self.toggle_recording)
        bottom_layout.addWidget(self.voice_btn)
        
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("پیام خود را بنویسید...")
        self.message_input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ccc;
                border-radius: 15px;
                padding: 15px;
                font-size: 16px;
                margin-left: 10px;
                margin-right: 10px;
            }
        """)
        bottom_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("⮞")
        self.send_button.setFixedSize(100, 50)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 15px;
                padding: 15px;
                font-size: 28px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
        """)
        bottom_layout.addWidget(self.send_button)
        
        layout.addLayout(bottom_layout)
        
        self.send_button.clicked.connect(self.send_message)
        
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.update_typing_animation)
        self.typing_dots = 0
        
    def add_message(self, text, is_user=True):
       
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        
        message = AnimatedLabel(text)
        message.setWordWrap(True)
        message.setStyleSheet(f"""
            QLabel {{
                background-color: {'#e3f2fd' if is_user else '#f5f5f5'};
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                font-size: 16px;
                max-width: 600px;
                transition: all 0.3s ease;
            }}
        """)
        
        align_layout = QHBoxLayout()
        if is_user:
            align_layout.addStretch()
            align_layout.addWidget(message)
        else:
            align_layout.addWidget(message)
            align_layout.addStretch()
            
        self.chat_layout.addLayout(align_layout)
        self.chat_layout.addStretch()
        
        message.fade_in()
        
        QTimer.singleShot(100, lambda: self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        ))
    
    def show_typing_indicator(self):
       
        self.typing_dots = 0
        self.typing_label = AnimatedLabel("در حال نوشتن")
        self.typing_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                font-size: 16px;
                color: #666;
            }
        """)
        
        align_layout = QHBoxLayout()
        align_layout.addWidget(self.typing_label)
        align_layout.addStretch()
        
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        self.chat_layout.addLayout(align_layout)
        self.chat_layout.addStretch()
        
        self.typing_label.fade_in()
        
        self.typing_timer.start(500)
    
    def hide_typing_indicator(self):
       
        def remove_typing():
            self.typing_timer.stop()
           
            for i in reversed(range(self.chat_layout.count())):
                item = self.chat_layout.itemAt(i)
                if item and item.widget() == self.typing_label:
                    self.chat_layout.takeAt(i)
                    item.widget().deleteLater()
                    break
        
        self.typing_label.fade_out()
        QTimer.singleShot(300, remove_typing)
    
    def send_message(self):
 
        text = self.message_input.toPlainText().strip()
        if text:
            
            original_style = self.send_button.styleSheet()
            self.send_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    border-radius: 15px;
                    padding: 15px;
                    font-size: 18px;
                }
            """)
            QTimer.singleShot(200, lambda: self.send_button.setStyleSheet(original_style))
            
            self.message_input.clear()
            
            self.add_message(text, True)
            
            QTimer.singleShot(300, self.show_typing_indicator)
            
            QTimer.singleShot(400, lambda: self.coach_manager.get_response(text, self.handle_response))
    
    def toggle_recording(self):
      
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        
        self.is_recording = True
        self.voice_btn.setText("■")
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 5px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        self.add_status_message("در حال ضبط صدا...")
        
        self.frames = []
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        
    def stop_recording(self):
        
        if not self.frames:
            self.add_error_message("صدایی ضبط نشد")
            return
            
        self.is_recording = False
        self.voice_btn.setText("⚫")
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 5px;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            self.add_status_message("در حال پردازش صدا...")
            
            
            timestamp = int(time.time())
            temp_filename = f'audio_{timestamp}.wav'
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            
           
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(4)  
            wf.setframerate(self.RATE)
            
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.float32)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            
            with wave.open(temp_path, 'rb') as wf:
                duration = wf.getnframes() / float(wf.getframerate())
                if duration < 0.5:  
                    self.add_error_message("صدای ضبط شده خیلی کوتاه است")
                    os.remove(temp_path)
                    return
                elif duration > 60: 
                    self.add_error_message("صدای ضبط شده خیلی طولانی است")
                    os.remove(temp_path)
                    return
            
            self.process_audio(temp_path)
            
        except Exception as e:
            self.add_error_message(f"خطا در ضبط صدا: {str(e)}")
            print(f"Error in recording: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self, audio_file):
        
        try:
            self.add_status_message("در حال تبدیل صدا به متن...")
            
            with open(audio_file, 'rb') as audio:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="fa",
                    temperature=0.3,  
                    prompt="این یک پیام صوتی فارسی است." 
                )
            
            try:
                os.remove(audio_file)
            except:
                pass
            
            if transcript.text:
            
                text = transcript.text.strip()
                self.message_input.setText(text)
                self.add_status_message(" صدا با موفقیت به متن تبدیل شد")
            else:
                self.add_error_message("متنی تشخیص داده نشد")
                
        except Exception as e:
            self.add_error_message(f"خطا در تبدیل صدا به متن: {str(e)}")
            print(f"Error in speech to text: {str(e)}")
            try:
                os.remove(audio_file)
            except:
                pass
    
    def handle_response(self, response):
        
        self.hide_typing_indicator()
        
        QTimer.singleShot(300, lambda: self.show_response(response))
        
    def show_response(self, response):
        
        if response:
            self.add_message(response, False)
        else:
            error_text = "متأسفانه در دریافت پاسخ مشکلی پیش آمد"
            self.add_message(error_text, False)

    def update_typing_animation(self):
        
        self.typing_dots = (self.typing_dots + 1) % 4
        dots = "." * self.typing_dots
        new_text = f"در حال نوشتن{dots}"
        
        self.typing_label.fade_out()
        QTimer.singleShot(150, lambda: self.typing_label.setText(new_text))
        QTimer.singleShot(150, self.typing_label.fade_in)

    def add_status_message(self, text):
        
        status_label = QLabel(text)
        status_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                padding: 5px;
                margin: 5px;
                font-size: 14px;
                font-style: italic;
            }
        """)
        status_label.setAlignment(Qt.AlignCenter)
        
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        self.chat_layout.addWidget(status_label)
        self.chat_layout.addStretch()
    
    def add_error_message(self, text):
        
        error_label = QLabel(text)
        error_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                padding: 5px;
                margin: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        error_label.setAlignment(Qt.AlignCenter)
        
        self.chat_layout.takeAt(self.chat_layout.count() - 1)
        self.chat_layout.addWidget(error_label)
        self.chat_layout.addStretch()
