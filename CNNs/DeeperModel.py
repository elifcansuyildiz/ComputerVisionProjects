"""
Elif Cansu YILDIZ 06/2021
"""
import torch.nn as nn

class DeeperModel(nn.Module):
    def __init__(self, batchNorm=False):
        
        super(DeeperModel, self).__init__()
        
        self.name = "Deeper Model (BatchNorm=" + str(batchNorm) + ")"
        self.savename = "model_deeper_batchnorm_" + str(batchNorm) + ".pt"
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=3, stride=1, padding=0)
        batch1 = nn.BatchNorm2d(10)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(in_channels=10, out_channels=20,  kernel_size=3, stride=1, padding=0)
        batch2 = nn.BatchNorm2d(20)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(in_channels=20, out_channels=40,  kernel_size=3, stride=1, padding=0)
        batch3 = nn.BatchNorm2d(40)
        relu3 = nn.ReLU()
        conv4 = nn.Conv2d(in_channels=40, out_channels=80,  kernel_size=3, stride=1, padding=0)
        batch4 = nn.BatchNorm2d(80)
        relu4 = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if batchNorm:
            self.conv_layers = nn.Sequential(conv1, batch1, relu1, conv2, batch2, relu2, maxpool, conv3, batch3, relu3, conv4, batch4, relu4, maxpool)
            
        else:
            self.conv_layers = nn.Sequential(conv1, relu1, conv2, relu2, maxpool, conv3, relu3, conv4, relu4, maxpool)
            
        # fully connected classifier
        in_dim = 80 * 4 * 4
        self.fc1 = nn.Linear(in_features=in_dim, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=10)
    
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.conv_layers(x)
        out1_flat = out1.view(cur_b_size, -1)
        out2 = self.fc1(out1_flat)
        y = self.fc2(out2)
        return y
