import sys
from PyQt5.QtWidgets import QApplication
from ui.home_window import HomeWindow
from core.coach_manager import CoachManager
from processes.exercise_detector import ExerciseDetector


def main():
    app = QApplication(sys.argv)
    detector = ExerciseDetector()
    window = HomeWindow(detector)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 