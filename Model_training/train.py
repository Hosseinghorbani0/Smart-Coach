import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import SportNet
from dataset import WorkoutDataset
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import logging
import random
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class CustomTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.phase_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.form_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.cycle_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'phase_acc': [],
            'form_acc': [],
            'cycle_acc': []
        }
        
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.grad_clip_value = config.get('grad_clip_value', 1.0)
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_augmentation = config.get('use_augmentation', True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        phase_correct = 0
        form_correct = 0
        cycle_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (videos, phase_labels, form_labels, cycle_labels) in enumerate(progress_bar):
            videos = videos.to(self.device)
            phase_labels = phase_labels.to(self.device)
            form_labels = form_labels.to(self.device)
            cycle_labels = cycle_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                phase_pred, form_pred, cycle_pred = self.model(videos)
                
                phase_loss = self.phase_criterion(phase_pred, phase_labels)
                form_loss = self.form_criterion(form_pred, form_labels)
                cycle_loss = self.cycle_criterion(cycle_pred, cycle_labels)
                
                loss = (0.4 * phase_loss + 0.3 * form_loss + 0.3 * cycle_loss)
            
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            phase_correct += (phase_pred.argmax(dim=1) == phase_labels).sum().item()
            form_correct += (form_pred.argmax(dim=1) == form_labels).sum().item()
            cycle_correct += (cycle_pred.argmax(dim=1) == cycle_labels).sum().item()
            total_samples += videos.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'phase_acc': f'{100 * phase_correct / total_samples:.2f}%',
                'form_acc': f'{100 * form_correct / total_samples:.2f}%',
                'cycle_acc': f'{100 * cycle_correct / total_samples:.2f}%'
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'phase_acc': 100 * phase_correct / total_samples,
            'form_acc': 100 * form_correct / total_samples,
            'cycle_acc': 100 * cycle_correct / total_samples
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        phase_correct = 0
        form_correct = 0
        cycle_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for videos, phase_labels, form_labels, cycle_labels in tqdm(self.val_loader, desc='Validation'):
                videos = videos.to(self.device)
                phase_labels = phase_labels.to(self.device)
                form_labels = form_labels.to(self.device)
                cycle_labels = cycle_labels.to(self.device)
                
                phase_pred, form_pred, cycle_pred = self.model(videos)
                
                phase_loss = self.phase_criterion(phase_pred, phase_labels)
                form_loss = self.form_criterion(form_pred, form_labels)
                cycle_loss = self.cycle_criterion(cycle_pred, cycle_labels)
                
                loss = (0.4 * phase_loss + 0.3 * form_loss + 0.3 * cycle_loss)
                
                total_loss += loss.item()
                phase_correct += (phase_pred.argmax(dim=1) == phase_labels).sum().item()
                form_correct += (form_pred.argmax(dim=1) == form_labels).sum().item()
                cycle_correct += (cycle_pred.argmax(dim=1) == cycle_labels).sum().item()
                total_samples += videos.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'phase_acc': 100 * phase_correct / total_samples,
            'form_acc': 100 * form_correct / total_samples,
            'cycle_acc': 100 * cycle_correct / total_samples
        }
    
    def train(self):
        logging.info("Starting model training...")
        
        for epoch in range(self.config['epochs']):
            logging.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            train_metrics = self.train_epoch()
            logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Phase Acc: {train_metrics['phase_acc']:.2f}%, "
                        f"Form Acc: {train_metrics['form_acc']:.2f}%, "
                        f"Cycle Acc: {train_metrics['cycle_acc']:.2f}%")
            
            val_metrics = self.validate()
            logging.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                        f"Phase Acc: {val_metrics['phase_acc']:.2f}%, "
                        f"Form Acc: {val_metrics['form_acc']:.2f}%, "
                        f"Cycle Acc: {val_metrics['cycle_acc']:.2f}%")
            
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['phase_acc'].append(val_metrics['phase_acc'])
            self.training_history['form_acc'].append(val_metrics['form_acc'])
            self.training_history['cycle_acc'].append(val_metrics['cycle_acc'])
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
                
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                
                self.best_model_path = os.path.join(
                    self.config['model_dir'],
                    f'best_model_epoch_{epoch + 1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }, self.best_model_path)
                logging.info(f"Best model saved: {self.best_model_path}")
            else:
                self.early_stopping_counter += 1
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                logging.info("Early stopping activated!")
                break
        
        self.save_training_history()
        self.plot_training_history()
        
        logging.info("Training completed!")
    
    def save_training_history(self):
        history_path = os.path.join(self.config['model_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
        logging.info(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history['phase_acc'], label='Phase Accuracy')
        plt.title('Phase Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history['form_acc'], label='Form Accuracy')
        plt.title('Form Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.training_history['cycle_acc'], label='Cycle Accuracy')
        plt.title('Cycle Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_dir'], 'training_history.png'))
        plt.close()
        logging.info("Training plots saved")

def main():
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 100,
        'early_stopping_patience': 10,
        'grad_clip_value': 1.0,
        'use_augmentation': True,
        'model_dir': 'saved_models',
        'data_dir': 'data'
    }
    
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    train_dataset = WorkoutDataset(
        root_dir=os.path.join(config['data_dir'], 'train'),
        transform=None
    )
    val_dataset = WorkoutDataset(
        root_dir=os.path.join(config['data_dir'], 'val'),
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = SportNet()
    trainer = CustomTrainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main() 