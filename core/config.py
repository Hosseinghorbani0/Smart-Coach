import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


class Config:
    BASE_DIR = Path(os.environ.get("SMART_COACH_BASE_DIR", Path(__file__).resolve().parent.parent))

    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")

    @classmethod
    def get_openai_api_key(cls):
        return os.environ.get("SMART_COACH_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    OPENAI_API_KEY = os.environ.get("SMART_COACH_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    COACH_PERSONA = """
    شما یک مربی هوشمند و صمیمی برای تمرینات بدنسازی هستید.
    به کاربر با احترام، صادق و دوستانه پاسخ بده. اگر چیزی را نمی‌دانید، صریح بگویید و از حدس زدن جلوگیری کن.
    تمرکز اصلی شما روی 4 حرکت اسکات، ددلیفت، پوش‌آپ و پول‌آپ است.
    در پاسخ‌ها فقط روی این حرکات تمرکز کن و از پیشنهادهای نامرتبط دوری کن.
    نام شما نیکوت است. اگر کسی نام شما را پرسید، بگویید «نیکوت هستم».
    سازنده شما حسین قربانی است و اگر درباره او سؤال شد، اطلاعات کوتاه و واقع‌بینانه بده.
    ایمیل: hosseingh1068@gmail.com
    وب‌سایت: hosseinghorbani0.ir
    """

    AUDIO_SETTINGS = {
        'sample_rate': 44100,
        'channels': 1,
        'format': 'wav'
    }

    STORAGE = {
        'voice_recordings': BASE_DIR / 'storage' / 'voice',
        'chat_history': BASE_DIR / 'storage' / 'chat'
    }

    @classmethod
    def ensure_storage_dirs(cls, base_dir=None):
        root = Path(base_dir) if base_dir else cls.BASE_DIR
        storage_root = root / 'storage'
        for relative_path in ('voice', 'chat'):
            (storage_root / relative_path).mkdir(parents=True, exist_ok=True)
        return storage_root

    @classmethod
    def init_storage(cls):
        cls.ensure_storage_dirs()
        for path in cls.STORAGE.values():
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_storage_paths(cls):
        cls.init_storage()
        return cls.STORAGE
