import os
import json
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check MediaPipe availability gracefully
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe is not installed. Keypoint extraction will operate in fallback zero-tensor mode.")


class VideoPreprocessor:
    """
    Applies spatial data augmentations, uniform image resizing, and ImageNet normalization to video frames.
    Uses Albumentations for accelerated image transformations.
    """
    def __init__(self, is_train: bool = True, image_size: Tuple[int, int] = (224, 224)):
        self.is_train = is_train
        self.image_size = image_size
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
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
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Applies transformation pipeline to a single RGB frame."""
        return self.transform(image=image)['image']


class WorkoutDataset(Dataset):
    """
    Production-ready PyTorch Dataset for multi-task workout video analysis.
    Extracts temporally aligned RGB frame sequences, 3D MediaPipe pose keypoints,
    and frame-level phase, form, and repetition cycle labels.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[VideoPreprocessor] = None,
        sequence_length: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        cache_keypoints: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else VideoPreprocessor(is_train=True, image_size=image_size)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.cache_keypoints = cache_keypoints
        
        # Lazy MediaPipe instantiation to avoid PyTorch DataLoader worker pickling crashes
        self._pose_model = None
        
        # In-memory keypoint cache for accelerated GPU training
        self._keypoint_cache: Dict[str, np.ndarray] = {}

        self.samples: List[Dict[str, Any]] = []
        self._scan_videos()
        
        logger.info(f"Dataset loaded from {root_dir}")
        logger.info(f"Total valid samples: {len(self.samples)} | Sequence length: {sequence_length}")

    def _scan_videos(self) -> None:
        """Scans dataset directory for .mp4 videos and matching metadata JSON files."""
        if not self.root_dir.exists():
            logger.error(f"Directory {self.root_dir} does not exist.")
            return

        for exercise_dir in sorted(self.root_dir.glob('*')):
            if not exercise_dir.is_dir():
                continue
                
            for video_path in sorted(exercise_dir.glob('*.mp4')):
                json_path = video_path.with_suffix('.json')
                if not json_path.exists():
                    logger.warning(f"Missing metadata JSON for video: {video_path}")
                    continue
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    self.samples.append({
                        'video_path': str(video_path),
                        'exercise_type': exercise_dir.name,
                        'phase_labels': metadata.get('phase_labels', []),
                        'form_labels': metadata.get('form_labels', []),
                        'cycle_labels': metadata.get('cycle_labels', [])
                    })
                except Exception as e:
                    logger.error(f"Failed to parse metadata file {json_path}: {e}")

    def _get_pose_model(self):
        """
        Lazily initializes MediaPipe Pose model per thread/process worker.
        Prevents multiprocessing serialization/pickling crashes in DataLoader.
        """
        if self._pose_model is None and MEDIAPIPE_AVAILABLE:
            self._pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Balanced for speed & accuracy
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._pose_model

    def _sample_frames(self, video_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Reads video and samples evenly spaced frames across the timeline.
        Returns extracted RGB frames and corresponding original frame indices.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Unable to open video file: {video_path}")
            dummy_frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            return [dummy_frame] * self.sequence_length, np.linspace(0, self.sequence_length - 1, self.sequence_length, dtype=int)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.sequence_length

        # Compute uniform temporal sample indices
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Black frame fallback if seeking fails
                dummy = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                frames.append(dummy)

        cap.release()
        return frames, indices

    def _extract_keypoints(self, frames: List[np.ndarray], video_path: str) -> np.ndarray:
        """
        Extracts 3D pose landmarks (x, y, z, visibility) for 33 MediaPipe keypoints.
        Supports in-memory caching to prevent redundant CPU computation across epochs.
        """
        if self.cache_keypoints and video_path in self._keypoint_cache:
            return self._keypoint_cache[video_path]

        pose_model = self._get_pose_model()
        pose_features = []

        for frame in frames:
            if pose_model is not None:
                results = pose_model.process(frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    features = []
                    for lm in landmarks:
                        features.extend([lm.x, lm.y, lm.z, lm.visibility])
                    pose_features.append(features)
                else:
                    pose_features.append([0.0] * (33 * 4))
            else:
                pose_features.append([0.0] * (33 * 4))

        keypoints_array = np.array(pose_features, dtype=np.float32)
        
        if self.cache_keypoints:
            self._keypoint_cache[video_path] = keypoints_array

        return keypoints_array

    def _align_labels(self, raw_labels: List[int], indices: np.ndarray) -> torch.Tensor:
        """
        Subsamples raw video frame labels using the exact frame indices
        sampled during video reading to ensure exact spatial-temporal alignment.
        """
        if not raw_labels:
            return torch.zeros(self.sequence_length, dtype=torch.long)

        aligned = []
        num_labels = len(raw_labels)
        for idx in indices:
            safe_idx = min(idx, num_labels - 1)
            aligned.append(raw_labels[safe_idx])

        return torch.tensor(aligned, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        video_path = sample['video_path']

        # 1. Sample temporally aligned RGB frames and frame indices
        frames, indices = self._sample_frames(video_path)

        # 2. Extract pose keypoints
        pose_features = self._extract_keypoints(frames, video_path)

        # 3. Apply transformations and stack frames into standard (T, C, H, W) tensor
        transformed_frames = torch.stack([self.transform(frame) for frame in frames])

        # 4. Convert pose keypoints tensor -> shape (T, 132)
        pose_tensor = torch.tensor(pose_features, dtype=torch.float32)

        # 5. Align labels to match sequence length exactly
        phase_labels = self._align_labels(sample['phase_labels'], indices)
        form_labels = self._align_labels(sample['form_labels'], indices)
        cycle_labels = self._align_labels(sample['cycle_labels'], indices)

        return {
            'frames': transformed_frames,              # Shape: [T, C, H, W]
            'pose_features': pose_tensor,             # Shape: [T, 132]
            'phase_labels': phase_labels,             # Shape: [T]
            'form_labels': form_labels,               # Shape: [T]
            'cycle_labels': cycle_labels              # Shape: [T]
        }


def _worker_init_fn(worker_id: int):
    """Ensures distinct NumPy and PyTorch random seeds across DataLoader workers."""
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def setup_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 16,
    sequence_length: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates robust DataLoader instances for training and validation splits.
    
    Args:
        root_dir: Root dataset directory containing 'train' and 'val' folders.
        batch_size: Batch size per iteration.
        sequence_length: Target frame sequence length.
        image_size: Target (height, width) for image frames.
        num_workers: Number of parallel multiprocessing data loader workers.
    """
    root_path = Path(root_dir)

    train_dataset = WorkoutDataset(
        root_dir=root_path / 'train',
        transform=VideoPreprocessor(is_train=True, image_size=image_size),
        sequence_length=sequence_length,
        image_size=image_size
    )

    val_dataset = WorkoutDataset(
        root_dir=root_path / 'val',
        transform=VideoPreprocessor(is_train=False, image_size=image_size),
        sequence_length=sequence_length,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn
    )

    return train_loader, val_loader
