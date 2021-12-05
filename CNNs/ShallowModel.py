"""
Elif Cansu YILDIZ 06/2021
"""
import torch.nn as nn

class ShallowModel(nn.Module):
    def __init__(self):
        
        super(ShallowModel, self).__init__()
        
        self.name = "Shallow Model"
        self.savename = "model_shallow.pt"
    
        conv1 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(in_channels=10, out_channels=20,  kernel_size=3, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layers = nn.Sequential(conv1, relu1, conv2, relu2, maxpool)

        # fully connected classifier
        in_dim = 20 * 12 * 12
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
                                
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.conv_layers(x)
        out1_flat = out1.view(cur_b_size, -1)
        y = self.fc(out1_flat)
        return y
