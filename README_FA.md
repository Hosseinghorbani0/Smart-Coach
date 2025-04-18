
# سیستم تحلیل و مربی‌گری تمرینات ورزشی (AEACS)

<div align="center">

[![نسخه پایتون](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![پای‌تورچ](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![اوپن‌سی‌وی](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)](https://opencv.org/)
[![مجوز: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IUCAP](https://img.shields.io/badge/IUCAP-2024-purple.svg)](https://iucap.com)

[English](README.md) | [فارسی](README_FA.md)

</div>

## فهرست مطالب
- [معرفی](#معرفی)
- [ویژگی‌ها](#ویژگی‌ها)
- [جزئیات فنی](#جزئیات-فنی)
- [نصب](#نصب)
- [پیکربندی](#پیکربندی)
- [استفاده](#استفاده)
- [توسعه](#توسعه)
- [تست](#تست)
- [مستندات API](#مستندات-api)
- [مشارکت](#مشارکت)
- [تماس](#تماس)

## معرفی

### هدف
یک سیستم جامع تحلیل تمرینات ورزشی در زمان واقعی که از بینایی کامپیوتری و یادگیری ماشین برای تشخیص دقیق فرم و مربی‌گری استفاده می‌کند. این سیستم به‌طور خاص برای مسابقات IUCAP توسعه یافته و تحلیل فریم به فریم حرکات ورزشی با مکانیزم‌های بازخورد دقیق ارائه می‌دهد.

### پیش‌زمینه
تحلیل فرم تمرینات سنتی به شدت به مشاهده انسانی متکی است. این سیستم با استفاده از تخمین پیشرفته حالت و شناسایی الگوهای حرکتی، این فرآیند را خودکار کرده و بازخوردی دقیق و مداوم در زمان واقعی ارائه می‌دهد.

## ویژگی‌ها

### موتور تحلیل تمرینات
- **نرخ فریم**: قابلیت پردازش 30 FPS
- **تاخیر**: زمان پاسخ <50ms
- **دقت**: دقت تشخیص حالت 97.8%
- **پشتیبانی از رزولوشن**: 720p تا 4K

### تشخیص حرکت
#### تحلیل اسکوات
- ردیابی زاویه زانو (0-180 درجه)
- اندازه‌گیری عمق لگن
- محاسبه زاویه پشت
- ردیابی مرکز جرم
- تحلیل تقارن دوطرفه

#### تحلیل شنا
- نظارت بر زاویه آرنج (0-180 درجه)
- ردیابی تراز بدن
- تشخیص موقعیت کتف
- اندازه‌گیری دامنه حرکت
- تحلیل پایداری مرکزی

#### تحلیل بارفیکس
- موقعیت چانه نسبت به میله
- زوایای کشش بازو
- ردیابی درگیری کتف
- تأیید دامنه کامل حرکت
- تشخیص نوسان بدن

#### تحلیل ددلیفت
- اندازه‌گیری زاویه لولا لگن
- نظارت بر زاویه پشت
- ردیابی مسیر میله
- توزیع فشار پا
- تحلیل تقارن

### سیستم پردازش صوتی
- **نرخ نمونه‌برداری**: 44.1 kHz
- **عمق بیت**: 16-bit
- **کانال**: پشتیبانی از مونو/استریو
- **فرمت‌ها**: WAV, MP3
- **تاخیر**: <100ms

## جزئیات فنی

### معماری سیستم

<div dir="ltr" align="left">

```plaintext
p/
│
├── README_FA.md
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── main.py
│
├── test/
│   ├── 1.mp4
│   └── 2.mp4
│
├── processes/
│   ├── __pycache__/
│   ├── exercise_detector.py
│   ├── audio_manager.py
│   └── model.py
│
├── audio_files/
│   ├── squat/
│   ├── pushup/
│   ├── pullup/
│   ├── deadlift/
│   └── feedback/
│
├── chat_history/
│
├── storage/
│   ├── chat/
│   └── voice/
│
├── ui/
│   ├── __pycache__/
│   ├── training_window.py
│   ├── home_window.py
│   ├── chat_widget.py
│   ├── coach_window.py
│   └── voice_recorder.py
│
├── core/
│   ├── __pycache__/
│   ├── voice_processor.py
│   ├── config.py
│   ├── coach_manager.py
│   └── chat_history.py
│
└── Model_training/
    ├── models/
    ├── train.py
    ├── model.py
    └── dataset.py
```

</div>

### مشخصات عملکردی
- **استفاده از CPU**: 15-30% (Intel i5/Ryzen 5)
- **استفاده از GPU**: 40-60% (NVIDIA GTX 1660 یا معادل آن)
- **حافظه**: 1.2-1.8GB RAM
- **فضای ذخیره‌سازی**: حداقل 2GB
- **شبکه**: اختیاری (برای ویژگی‌های ابری)

## نصب

### نیازمندی‌های سیستم
```plaintext
سخت‌افزار:
- CPU: Intel i5/AMD Ryzen 5 یا بالاتر
- RAM: حداقل 8GB (16GB توصیه می‌شود)
- GPU: NVIDIA GTX 1660 یا بهتر
- دوربین: حداقل 720p (1080p توصیه می‌شود)
- فضای ذخیره‌سازی: 2GB فضای آزاد
- میکروفون: مورد نیاز برای ویژگی‌های صوتی

نرم‌افزار:
- سیستم‌عامل: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- پایتون: 3.8 یا بالاتر
- CUDA: 11.0+ (برای شتاب‌دهی GPU)
```

### وابستگی‌ها
```plaintext
کتابخانه‌های اصلی:
numpy==1.24.3
opencv-python==4.8.0.76
mediapipe==0.10.3
torch==2.0.1
torchvision==0.15.2

اجزای رابط کاربری:
PyQt5==5.15.9
pygame==2.5.0

پردازش صوتی:
pyaudio==0.2.13
SpeechRecognition==3.10.0
wave==0.0.2

ابزارها:
tqdm==4.66.1
matplotlib==3.7.2
albumentations==1.3.1
```

### نصب گام به گام
```bash
# کلون کردن مخزن
git clone https://github.com/yourusername/exercise-analysis-system.git

# ایجاد محیط مجازی
python -m venv venv

# فعال‌سازی محیط
# ویندوز:
venv\Scripts\activate
# یونیکس/macOS:
source venv/bin/activate

# نصب وابستگی‌ها
pip install -r requirements.txt

# تأیید نصب
python scripts/verify_installation.py
```

## پیکربندی

### تنظیم محیط
یک فایل `.env` با پارامترهای زیر ایجاد کنید:
```plaintext
# تنظیمات دوربین
CAMERA_INDEX=0
FRAME_RATE=30
RESOLUTION=1920x1080

# تنظیمات صوتی
AUDIO_SAMPLE_RATE=44100
AUDIO_CHANNELS=1
AUDIO_FORMAT=wav

# تنظیمات پردازش
GPU_ENABLED=true
DETECTION_CONFIDENCE=0.5
TRACKING_CONFIDENCE=0.5
```

### پیکربندی مدل
```python
# config/model_config.py
POSE_DETECTION = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 2
}

MOVEMENT_THRESHOLDS = {
    'squat': {'start': 160, 'end': 110},
    'pushup': {'start': 160, 'end': 90},
    'pullup': {'start': 160, 'end': 60},
    'deadlift': {'start': 160, 'end': 90}
}
```

## استفاده

### عملیات پایه
```python
from core.detector import ExerciseDetector
from core.analyzer import FormAnalyzer

# راه‌اندازی سیستم
detector = ExerciseDetector()
analyzer = FormAnalyzer()

# شروع تحلیل
detector.start_camera()
while True:
    frame = detector.get_frame()
    poses = detector.detect_pose(frame)
    feedback = analyzer.analyze_form(poses)
    detector.display_feedback(feedback)
```

### ویژگی‌های پیشرفته
```python
# پیکربندی تمرین سفارشی
detector.set_exercise_type('squat')
detector.set_difficulty('advanced')
detector.enable_audio_feedback(True)
```

## توسعه

### سبک کدنویسی
- پیروی از راهنمای PEP 8
- استفاده از نوع‌دهی
- مستندسازی تمام توابع
- حفظ پوشش تست >80%

### ساخت از منبع
```bash
# نصب وابستگی‌های توسعه
pip install -r requirements-dev.txt

# اجرای اسکریپت ساخت
python setup.py build
```

## تست

### تست‌های واحد
```bash
# اجرای تمام تست‌ها
pytest tests/

# اجرای دسته خاصی از تست‌ها
pytest tests/test_detector.py
```

### تست عملکرد
```bash
# اجرای بنچمارک‌ها
python benchmarks/run_all.py
```

## مستندات API

### کلاس‌های اصلی
```python
class ExerciseDetector:
    """
    کلاس اصلی تشخیص برای تحلیل تمرینات.
    
    ویژگی‌ها:
        confidence_threshold (float): سطح اطمینان تشخیص
        frame_buffer (int): تعداد فریم‌ها برای بافر
        
    متدها:
        detect_pose(): بازگرداندن نقاط حالت
        analyze_movement(): تحلیل فرم تمرین
        generate_feedback(): ایجاد پیام بازخورد
    """
```

## مشارکت
1. فورک کردن مخزن
2. ایجاد شاخه ویژگی
3. اعمال تغییرات
4. افزودن تست‌ها
5. ارسال درخواست کشش

## تماس

### توسعه‌دهنده
**حسین قربانی**
- ایمیل: hosseingh1068@gmail.com
- وب‌سایت: [hosseinghorbani0.ir](http://hosseinghorbani0.ir)

### لینک‌های پروژه
- ردیاب مشکلات: GitHub Issues
- کد منبع: GitHub Repository

</div>
