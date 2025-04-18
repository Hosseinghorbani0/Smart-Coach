import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path
import json
import mediapipe as mp

class VideoPreprocessor:
    def __init__(self, is_train=True):
        self.is_train = is_train
        
        if is_train:
            self.transform = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.5),
                
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.3),
                
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03,
                        p=1.0
                    ),
                    A.GridDistortion(p=1.0),
                ], p=0.3),
                
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        return self.transform(image=image)['image']

class WorkoutDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=32):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else VideoPreprocessor(is_train=True)
        self.sequence_length = sequence_length
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.samples = []
        self._scan_videos()
        
        logging.info(f"Dataset loaded from {root_dir}")
        logging.info(f"Total samples: {len(self.samples)}")
        logging.info(f"Sequence length: {sequence_length}")
    
    def _scan_videos(self):
        for exercise_dir in self.root_dir.glob('*'):
            if not exercise_dir.is_dir():
                continue
                
            for video_path in exercise_dir.glob('*.mp4'):
                json_path = video_path.with_suffix('.json')
                if not json_path.exists():
                    continue
                
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                self.samples.append({
                    'video_path': str(video_path),
                    'exercise_type': exercise_dir.name,
                    'phase_labels': metadata['phase_labels'],
                    'form_labels': metadata['form_labels'],
                    'cycle_labels': metadata['cycle_labels']
                })
    
    def _sample_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _extract_keypoints(self, frames):
        pose_features = []
        
        for frame in frames:
            results = self.pose.process(frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = []
                
                for landmark in landmarks:
                    features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                pose_features.append(features)
            else:
                pose_features.append([0] * (33 * 4))
        
        return np.array(pose_features)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames = self._sample_frames(sample['video_path'])
        pose_features = self._extract_keypoints(frames)
        
        frames = torch.stack([self.transform(frame) for frame in frames])
        pose_features = torch.FloatTensor(pose_features)
        
        phase_labels = torch.LongTensor(sample['phase_labels'])
        form_labels = torch.LongTensor(sample['form_labels'])
        cycle_labels = torch.LongTensor(sample['cycle_labels'])
        
        return {
            'frames': frames,
            'pose_features': pose_features,
            'phase_labels': phase_labels,
            'form_labels': form_labels,
            'cycle_labels': cycle_labels
        }

def setup_dataloaders(root_dir, batch_size=32, num_workers=4):
    train_dataset = WorkoutDataset(
        root_dir=os.path.join(root_dir, 'train'),
        transform=VideoPreprocessor(is_train=True)
    )
    
    val_dataset = WorkoutDataset(
        root_dir=os.path.join(root_dir, 'val'),
        transform=VideoPreprocessor(is_train=False)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 