import sys

from core.config import Config


def main():
    try:
        from PyQt5.QtWidgets import QApplication
        from ui.home_window import HomeWindow
        from processes.exercise_detector import ExerciseDetector
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Install them with: python -m pip install -r requirements.txt"
        ) from exc

    Config.init_storage()

    app = QApplication(sys.argv)
    detector = ExerciseDetector()
    window = HomeWindow(detector)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 