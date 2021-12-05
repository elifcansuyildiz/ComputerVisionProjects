"""
Elif Cansu YILDIZ 06/2021
"""
import torch
import torch.nn as nn

class WiderModel(nn.Module):
    def __init__(self):
        
        super(WiderModel, self).__init__()
        
        self.name = "Wider Model"
        self.savename = "model_wider.pt"
    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_dim = 40 * 13 * 13
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
                                
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        x = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(cur_b_size, -1)
        x = self.fc(x)
        return x
