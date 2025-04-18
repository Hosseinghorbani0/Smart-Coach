import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import cv2

class CustomAttention(nn.Module):
    def __init__(self, in_channels):
        super(CustomAttention, self).__init__()
        self.spatial_conv = nn.Conv3d(in_channels, in_channels, 
                                    kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1))
        self.temporal_conv = nn.Conv3d(in_channels, in_channels, 
                                     kernel_size=(3, 1, 1),
                                     padding=(1, 0, 0))
        self.norm = nn.BatchNorm3d(in_channels)
        
    def forward(self, x):
        spatial_attn = torch.sigmoid(self.norm(self.spatial_conv(x)))
        x = x * spatial_attn
        
        temporal_attn = torch.sigmoid(self.temporal_conv(x))
        x = x * temporal_attn
        
        return x

class ExerciseModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ExerciseModel, self).__init__()
        
        self.features = nn.Sequential(
            
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_rate),
            nn.MaxPool3d(kernel_size=2),
            
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_rate),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_rate),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.attention = CustomAttention(128)
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.phase_classifier = nn.Linear(64, 3)
        self.form_classifier = nn.Linear(64, 2)
        self.cycle_classifier = nn.Linear(64, 2)
        
        self.movement_thresholds = {
            'squat': {
                'start': lambda lm: self.is_standing_for_squat(lm),
                'end': lambda lm: self.is_standing_for_squat(lm)
            },
            'deadlift': {
                'start': lambda lm: self.is_deadlift_position(lm),
                'end': lambda lm: self.is_standing(lm)
            },
            'pushup': {
                'start': lambda lm: self.is_pushup_position(lm),
                'end': lambda lm: self.is_pushup_position(lm)
            },
            'pullup': {
                'start': lambda lm: self.is_hanging(lm),
                'end': lambda lm: self.is_hanging(lm)
            }
        }
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.features(x)
        
        x = self.attention(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, -1, 2048)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  
        
        features = self.fc(x)
        
        return {
            'phase': self.phase_classifier(features),
            'form': self.form_classifier(features),
            'cycle': self.cycle_classifier(features)
        }
    
    def compute_loss(self, outputs, targets, weights={'phase': 0.3, 'form': 0.5, 'cycle': 0.2}):
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        losses = {}
        for key in outputs:
            loss = criterion(outputs[key], targets[key])
            losses[key] = (loss.mean() * weights[key])
        
        total_loss = sum(losses.values())
        
        return {
            'total_loss': total_loss,
            **losses
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            return {
                'phase': torch.softmax(outputs['phase'], dim=1),
                'form': torch.softmax(outputs['form'], dim=1),
                'cycle': torch.softmax(outputs['cycle'], dim=1)
            }

    def prepare_movement_input(self, frames):
        if len(frames) < 3:
            print("⚠ تعداد فریم‌ها کافی نیست")
            return None
        
        try:
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            key_frames = [frames[i] for i in indices]
            
            processed_frames = []
            for frame in key_frames:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                tensor = torch.from_numpy(frame).float()
                tensor = tensor.permute(2, 0, 1)
                processed_frames.append(tensor)
            
            input_tensor = torch.stack(processed_frames)
            input_tensor = input_tensor.permute(1, 0, 2, 3)
            input_tensor = input_tensor.unsqueeze(0)
            
            return input_tensor
            
        except Exception as e:
            print(f" خطا در آماده‌سازی ورودی: {str(e)}")
            return None 