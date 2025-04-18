import cv2
import mediapipe as mp
import torch
import numpy as np
from processes.model import ExerciseModel
import os
from .audio_manager import ExerciseAudioManager

class ExerciseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.squat_count = 0
        self.deadlift_count = 0
        self.pushup_count = 0
        self.pullup_count = 0
        
        self.is_collecting = False
        self.current_exercise = None
        self.movement_frames = []
        
        self.squat_model = None
        self.deadlift_model = None
        
        self.base_model = ExerciseModel()
        
        self.models_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  
            'Model_training',
            'models'
        )
        
        self.models = {}
        self.model_paths = {
            'squat': os.path.join(self.models_base_path, 'squat_best_model.pth'),
            'pushup': os.path.join(self.models_base_path, 'pushup_best_model.pth'),
            'pullup': os.path.join(self.models_base_path, 'pullup_best_model.pth'),
            'deadlift': os.path.join(self.models_base_path, 'deadlift_best_model.pth')
        }
        
        print(f"Models base path: {self.models_base_path}")
        for exercise, path in self.model_paths.items():
            print(f"Model path for {exercise}: {path}")
            print(f"File exists: {os.path.exists(path)}")
        
        self.load_models()
        
        self.movement_thresholds = {
            'squat': {'start': 160, 'end': 110},
            'pushup': {'start': 160, 'end': 90},
            'pullup': {'start': 160, 'end': 60},
            'deadlift': {'start': 160, 'end': 90}
        }
        
        self.rep_count = {
            'squat': 0,
            'pushup': 0,
            'pullup': 0,
            'deadlift': 0
        }
        
        self.in_rep = False
        self.buffer_size = 32
        self.frame_buffer = []
        self.set_count = 0
        
        self.key_angles = {
            'pushup': {'elbow': 90},
            'pullup': {'chin_height': 0.5},
            'deadlift': {'back_angle': 45},
            'squat': {'knee_angle': 90}
        }
        
        self.movement_threshold = {
            'squat': {'knee': 130},
            'pushup': {'elbow': 90},
            'pullup': {'chin': 0.7},
            'deadlift': {'back': 45}
        }
        
        self.angle_threshold = {
            'start': 170,
            'collect': 150,
            'bottom': 110
        }
        
        self.last_angles = {}
        self.angle_history = []
        self.min_frames = 10
        self.max_frames = 100
        
        self.squat_conditions = {
            'knee_angle': False,
            'back_angle': False,
            'view_angle': False
        }
        
        self.deadlift_conditions = {
            'knee_angle': False,
            'back_angle': False,
            'hip_angle': False
        }
        
        self.pushup_conditions = {
            'body_position': False,
            'shoulder_angle': False,
            'elbow_angle': False
        }
        
        self.pullup_conditions = {
            'body_position': False,
            'elbow_angle': False,
            'chin_height': False
        }

        self.audio_manager = ExerciseAudioManager()

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def detect_exercise(self, landmarks):
        elbow_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        )
        
        knee_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        
        if abs(elbow_angle - self.key_angles['pushup']['elbow']) < 10:
            return 'pushup'
        elif knee_angle < 100:
            return 'squat'
        
        return None
    
    def calculate_score(self, angles):
        knee_angle = angles['knee']
        
        if knee_angle < 90:
            return 0.7
        elif knee_angle < 110:
            return 1.0
        elif knee_angle < 130:
            return 0.8
        else:
            return 0.5
    
    def process_frame(self, frame, auto_detect=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            angles = self.detect_key_angles(results.pose_landmarks.landmark)
            
            if angles:
                if (angles['knee'] > 165 and not self.is_collecting):
                    self.reset_conditions()
                
                if auto_detect:
                    if self.is_collecting:
                        self.movement_frames.append(frame.copy())
                        
                        if self.should_end_collection(angles):
                            movement_verified = self.verify_movement(self.current_exercise)
                            
                            if movement_verified:
                                self.rep_count[self.current_exercise] += 1
                                current_count = self.rep_count[self.current_exercise]
                                self.audio_manager.play_count(
                                    self.current_exercise, 
                                    current_count
                                )
                            
                            elif self.current_exercise is not None:
                                self.audio_manager.play_random_wrong_form()
                            
                            self.reset_conditions()
                            self.is_collecting = False
                            self.current_exercise = None
                            self.movement_frames = []
                            
                            return frame, None, self.rep_count
                        
                        return frame, f"در حال تشخیص {self.current_exercise}", self.rep_count
                    
                    if self.check_squat_conditions(angles):
                        self.is_collecting = True
                        self.current_exercise = 'squat'
                        self.movement_frames = [frame.copy()]
                        return frame, "شروع اسکات", self.rep_count
                    
                    elif self.check_deadlift_conditions(angles):
                        self.is_collecting = True
                        self.current_exercise = 'deadlift'
                        self.movement_frames = [frame.copy()]
                        return frame, "شروع ددلیفت", self.rep_count
                    
                    elif self.check_pushup_conditions(angles):
                        self.is_collecting = True
                        self.current_exercise = 'pushup'
                        self.movement_frames = [frame.copy()]
                        return frame, "شروع پوش‌آپ", self.rep_count
                    
                    elif self.check_pullup_conditions(angles):
                        self.is_collecting = True
                        self.current_exercise = 'pullup'
                        self.movement_frames = [frame.copy()]
                        return frame, "شروع پول‌آپ", self.rep_count
            
            self.draw_feedback(frame, results.pose_landmarks, angles, self.rep_count)
        
        return frame, None, self.rep_count

    def should_start_collection(self, angles):
        knee_angle = angles['knee']
        
        if knee_angle < 170 and not self.is_collecting:
            return True
        return False

    def should_end_collection(self, angles):
        if len(self.movement_frames) >= self.max_frames:
            return True
        
        if len(self.movement_frames) < self.min_frames:
            return False
        
        if self.current_exercise == 'squat' or self.current_exercise == 'deadlift':
            return angles['knee'] > 165
        
        elif self.current_exercise == 'pushup':
            return angles['elbow'] > 160
        
        elif self.current_exercise == 'pullup':
            return angles['chin_to_shoulder'] < -0.1
        
        return False

    def identify_exercise(self, frames):
        if len(frames) < 3:
            return None
        
        try:
            input_tensor = self.prepare_movement_input(frames)
            if input_tensor is None:
                return None
            
            results = {}
            
            for exercise_type, model in self.models.items():
                if model is None:
                    continue
                
                try:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        
                        phase_prob = torch.softmax(outputs['phase'], dim=1).max().item()
                        form_prob = torch.softmax(outputs['form'], dim=1)[0, 1].item()
                        cycle_prob = torch.softmax(outputs['cycle'], dim=1)[0, 1].item()
                        
                        total_score = (
                            cycle_prob * 0.4 +
                            phase_prob * 0.3 +
                            form_prob * 0.3
                        )
                        
                        results[exercise_type] = total_score
                
                except Exception as e:
                    continue
            
            if results:
                best_exercise = max(results.items(), key=lambda x: x[1])
                if best_exercise[1] > 0.6:
                    return best_exercise[0]
            
            return None
            
        except Exception as e:
            return None

    def detect_key_angles(self, landmarks):
        try:
            knee_angle = self.calculate_angle(
                [landmarks[23].x, landmarks[23].y],
                [landmarks[25].x, landmarks[25].y],
                [landmarks[27].x, landmarks[27].y]
            )
            
            elbow_angle = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y],
                [landmarks[13].x, landmarks[13].y],
                [landmarks[15].x, landmarks[15].y]
            )
            
            back_angle = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y],
                [landmarks[23].x, landmarks[23].y],
                [landmarks[25].x, landmarks[25].y]
            )
            
            hip_angle = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y],
                [landmarks[23].x, landmarks[23].y],
                [landmarks[25].x, landmarks[25].y]
            )
            
            chin_to_shoulder = landmarks[7].y - landmarks[11].y
            
            return {
                'knee': knee_angle,
                'elbow': elbow_angle,
                'back': back_angle,
                'hip': hip_angle,
                'chin_to_shoulder': chin_to_shoulder
            }
        
        except Exception as e:
            return None

    def calculate_back_angle(self, shoulder_left, shoulder_right, hip_left, hip_right):
        shoulder_mid = np.array([(shoulder_left.x + shoulder_right.x) / 2,
                               (shoulder_left.y + shoulder_right.y) / 2])
        hip_mid = np.array([(hip_left.x + hip_right.x) / 2,
                           (hip_left.y + hip_right.y) / 2])
        
        vertical = np.array([0, -1])
        back_vector = shoulder_mid - hip_mid
        
        angle = np.arccos(np.dot(vertical, back_vector) / 
                         (np.linalg.norm(vertical) * np.linalg.norm(back_vector)))
        return np.degrees(angle)

    def detect_exercise_type(self, angles):
        knee_angle = angles.get('knee', 180)
        hip_angle = angles.get('hip', 180)
        back_angle = angles.get('back', 90)
        
        if (hip_angle < 160 and back_angle > 45):
            return 'deadlift'
        elif knee_angle < 160:
            return 'squat'
        return None
    
    def count_repetition(self, angles):
        current_angle = angles['knee']
        
        if not self.in_rep and current_angle < self.movement_thresholds['squat']['end']:
            self.in_rep = True
            
        elif self.in_rep and current_angle > self.movement_thresholds['squat']['start']:
            self.rep_count += 1
            self.in_rep = False
    
    def draw_feedback(self, frame, landmarks, angles, rep_count):
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2)
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        for exercise, count in rep_count.items():
            if count > 0:
                cv2.putText(frame, f"{exercise}: {count}", (10, y_pos),
                           font, 0.7, (255, 255, 255), 2)
                y_pos += 30
        
        if self.is_collecting:
            cv2.putText(frame, "save", (10, y_pos),
                       font, 0.7, (0, 255, 255), 2)

    def prepare_input(self, frame):
        try:
            frame = cv2.resize(frame, (224, 224))
            
            tensor = torch.from_numpy(frame).float() / 255.0
            tensor = tensor.permute(2, 0, 1)
            
            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(2)
            
            tensor = tensor.repeat(1, 1, 3, 1, 1)
            
            return tensor
            
        except Exception as e:
            return None
    
    def _extract_features(self, landmarks):
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def _update_feedback(self, frame, landmarks, score):
        for landmark in landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.putText(frame, f"Reps: {self.rep_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def _evaluate_exercise(self):
        if len(self.frame_buffer) < self.buffer_size:
            return None
            
        with torch.no_grad():
            features = torch.tensor(self.frame_buffer).float()
            features = features.unsqueeze(0)  
            outputs = self.model(features)
            
            total_score = (
                outputs['form_score'].item() +
                outputs['angle_score'].item() +
                outputs['stability_score'].item()
            )
            
            return {
                'total_score': total_score,
                'form_score': outputs['form_score'].item(),
                'angle_score': outputs['angle_score'].item(),
                'stability_score': outputs['stability_score'].item(),
                'exercise_type': torch.argmax(outputs['exercise_type']).item()
            }

    
    def select_key_frames(self, frames):
        if len(frames) < 3:
            return frames
        
        mid_idx = len(frames) // 2
        return [frames[0], frames[mid_idx], frames[-1]]

    def prepare_movement_input(self, frames):
        try:
            if len(frames) > 32:
                indices = np.linspace(0, len(frames)-1, 32, dtype=int)
                key_frames = [frames[i] for i in indices]
            else:
                key_frames = frames
                while len(key_frames) < 32:
                    key_frames.extend(frames[:32-len(key_frames)])
            
            processed_frames = []
            for frame in key_frames:
                frame = cv2.resize(frame, (64, 64))
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame = frame / 255.0
                
                frame = (frame - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
                       np.array([0.229, 0.224, 0.225], dtype=np.float32)
                
                tensor = torch.from_numpy(frame).float()
                tensor = tensor.permute(2, 0, 1)
                processed_frames.append(tensor)
            
            input_tensor = torch.stack(processed_frames, dim=0)
            input_tensor = input_tensor.permute(1, 0, 2, 3)
            input_tensor = input_tensor.unsqueeze(0)
            
            expected_shape = (1, 3, 32, 64, 64)
            assert input_tensor.shape == expected_shape
            
            return input_tensor
            
        except Exception as e:
            return None

    def verify_movement(self, exercise_type):
        if not self.models[exercise_type] or len(self.movement_frames) < 3:
            return False
        
        try:
            input_tensor = self.prepare_movement_input(self.movement_frames)
            if input_tensor is None:
                return False
            
            with torch.no_grad():
                model = self.models[exercise_type]
                outputs = model(input_tensor)
                
                try:
                    cycle_prob = torch.softmax(outputs['cycle'], dim=1)[0, 1].item()
                    phase_prob = torch.softmax(outputs['phase'], dim=1).max().item()
                    form_quality = torch.softmax(outputs['form'], dim=1)[0, 1].item()
                    
                    total_score = (cycle_prob * 0.5 + phase_prob * 0.2 + form_quality * 0.3)
                    
                    return total_score > 0.6
                
                except Exception as e:
                    return False
        
        except Exception as e:
            return False

    def load_models(self):
        try:
            for exercise_type, model_path in self.model_paths.items():
                if os.path.exists(model_path):
                    print(f"Loading model for {exercise_type} from {model_path}")
                    
                    model = ExerciseModel()
                    try:
                        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                        model.load_state_dict(state_dict, strict=False)
                        model.eval()
                        self.models[exercise_type] = model
                        
                    except Exception as e:
                        print(f"Eror loding model for {exercise_type}: {str(e)}")
                        self.models[exercise_type] = None
                else:
                    print(f"Model file not : {model_path}")
                    self.models[exercise_type] = None
                    
        except Exception as e:
            print(f"Error in load_models: {str(e)}")

    def train_model(self, model, dataloaders, exercise_type):
        for videos, labels in dataloader:
            outputs = model(videos)

    def verify_model_directory(self):
        model_dir = os.path.join(os.path.dirname(__file__), 'modell')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir 

    def check_squat_conditions(self, angles):
        if 150 <= angles['hip'] <= 180:
            self.squat_conditions['view_angle'] = True
        
        if 75 <= angles['knee'] <= 120:
            self.squat_conditions['knee_angle'] = True
        
        if 150 <= angles['back'] <= 180:
            self.squat_conditions['back_angle'] = True
        
        if all(self.squat_conditions.values()):
            return True
        return False

    def check_deadlift_conditions(self, angles):
        if angles['knee'] < 80:
            self.deadlift_conditions['knee_angle'] = True
        
        if 110 <= angles['back'] <= 155:
            self.deadlift_conditions['back_angle'] = True
        
        if 100 <= angles['hip'] <= 130:
            self.deadlift_conditions['hip_angle'] = True
        
        if all(self.deadlift_conditions.values()):
            return True
        return False

    def check_pushup_conditions(self, angles):
        if 165 <= angles['back'] <= 180:
            self.pushup_conditions['body_position'] = True
        
        if 160 <= angles['hip'] <= 175:
            self.pushup_conditions['shoulder_angle'] = True
        
        if 75 <= angles['elbow'] <= 115:
            self.pushup_conditions['elbow_angle'] = True
        
        if all(self.pushup_conditions.values()):
            return True
        return False

    def check_pullup_conditions(self, angles):
        if 165 <= angles['back'] <= 180:
            self.pullup_conditions['body_position'] = True
        
        if 85 <= angles['elbow'] <= 110:
            self.pullup_conditions['elbow_angle'] = True
        
        if -0.2 <= angles['chin_to_shoulder'] <= 0.2:
            self.pullup_conditions['chin_height'] = True
        
        if all(self.pullup_conditions.values()):
            return True
        return False

    def reset_conditions(self):
        for condition in self.squat_conditions:
            self.squat_conditions[condition] = False
        for condition in self.deadlift_conditions:
            self.deadlift_conditions[condition] = False
        for condition in self.pushup_conditions:
            self.pushup_conditions[condition] = False
        for condition in self.pullup_conditions:
            self.pullup_conditions[condition] = False 