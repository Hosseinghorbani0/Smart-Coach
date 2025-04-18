import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AttnBlock(nn.Module):

    def __init__(self, in_channels):


        super(AttnBlock, self).__init__()
        
        self.space_conv = nn.Conv3d(in_channels, in_channels, 
                                    kernel_size=(1, 2, 2),
                                    padding=(0, 1, 1))
        
        self.time_conv = nn.Conv3d(in_channels, in_channels, 
                                     kernel_size=(2, 1, 1),
                                     padding=(1, 0, 0))
        
        self.norm = nn.BatchNorm3d(in_channels)

        
        self.final_conv = nn.Conv3d(in_channels, in_channels, 
                                   kernel_size=(1, 1, 1),
                                   padding=(0, 0, 0))
        
    def forward(self, x):


        s_att = torch.sigmoid(self.norm(self.space_conv(x)))
        x = x * s_att
        
        t_att = torch.sigmoid(self.time_conv(x))
        x = x * t_att
        
        x = self.final_conv(x)
        
        return x

class MovementNet(nn.Module):

    def __init__(self, input_size=(224, 224), num_frames=32):


        super(MovementNet, self).__init__()
        
        self.input_size = input_size
        self.num_frames = num_frames
        
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        self.attention = AttnBlock(256)
        
        with torch.no_grad():


            dummy_input = torch.zeros(1, 3, num_frames, input_size[0], input_size[1])

            dummy_output = self.backbone(dummy_input)

            self._feature_size = dummy_output.shape[1] * dummy_output.shape[3] * dummy_output.shape[4]
        
        self.lstm = nn.LSTM(
            input_size=self._feature_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.phase_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.form_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.cycle_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
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
                init.xavier_normal_(m.weight, gain=1.0)


                init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():

                    if 'weight' in name:
                        init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        init.zeros_(param)
    
    def forward(self, x):

        batch_size = x.size(0)
        
        x = self.backbone(x)
        
        x = self.attention(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, -1, self._feature_size)
        
        lstm_out, _ = self.lstm(x)
        
        features = self.fc(lstm_out[:, -1, :])
        
        return {
            'phase': self.phase_classifier(features),
            'form': self.form_classifier(features),
            'cycle': self.cycle_classifier(features)
        }
    
    def compute_loss(self, outputs, targets, weights={'phase': 0.3, 'form': 0.5, 'cycle': 0.2}):

        criterion = nn.CrossEntropyLoss()
        
        phase_loss = criterion(outputs['phase'], targets['phase'])
        form_loss = criterion(outputs['form'], targets['form'])
        cycle_loss = criterion(outputs['cycle'], targets['cycle'])
        
        total_loss = (0.3 * phase_loss + 
                     0.5 * form_loss + 
                     0.2 * cycle_loss)
        
        return {
            'total_loss': total_loss,
            'phase_loss': phase_loss,
            'form_loss': form_loss,
            'cycle_loss': cycle_loss
        }

def get_model_summary(model, input_size=(1, 3, 32, 224, 224)):
    return {
        'input_size': input_size,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

def main():
    params = {
        'sequence_length': 8,    
        'image_size': 64,     
        'batch_size': 4,      
        'num_epochs': 5,       
        'learning_rate': 0.001
    } 

class SportNet(nn.Module):

    def __init__(self, dropout_rate=0.3):

        super(SportNet, self).__init__()
        
        self.backbone = nn.Sequential(
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
        
        self.attention = AttnBlock(128)
        
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
        
        x = self.backbone(x)
        
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

def print_model_info(model):

    print("\n")
    print("\n")
    
    for name, param in model.named_parameters():
        print(f"{name}")
        print(f"{param.shape}")
        print(f"{param.dtype}")
        print(f"{' -> '.join(name.split('.'))}\n")
    
    print("\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,}")
    
    print("\n")
    print("- features")
    print("- lstm")
    print("- cycle_detector")
    print("- phase_detector")
    print("- form_detector")

if __name__ == "__main__":
    model = SportNet() 