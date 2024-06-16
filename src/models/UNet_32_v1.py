import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

class UNet_32_v1(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # Encoder
        self.e1 = nn.Conv2d(1, 32, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.e3 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Decoder
        self.d1 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.d2 = nn.ConvTranspose2d(128, 32, kernel_size=4, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.d3 = nn.ConvTranspose2d(64, 2, kernel_size=4, padding=1, stride=2)

        # Output layer
        self.outconv = nn.Conv2d(3, 2, kernel_size=3, padding=1, stride=1)



        self.device = params.device
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=params.learning_rate)


    def forward(self, x):
        # Encoder
        xe1 = F.relu(self.bn1(self.e1(x)))
        xe2 = F.relu(self.bn2(self.e2(xe1)))
        xe3 = F.relu(self.bn3(self.e3(xe2)))

        # Decoder
        xd1 = F.relu(self.bn4(self.d1(xe3)))
        xd1_cat = torch.cat([xd1, xe2], dim=1)
        
        xd2 = F.relu(self.bn5(self.d2(xd1_cat)))
        xd2_cat = torch.cat([xd2, xe1], dim=1)
        
        xd3 = self.d3(xd2_cat)

        # Concatenate the final output with the original input (assumed to be 1 channel)
        out = self.outconv(torch.cat([xd3, x], dim=1))

        return out
    
    def train_step(self, data):
        gray, color = data
        gray = gray.float().to(self.device)
        color = color.float().to(self.device)
        self.optimizer.zero_grad()
        output = self(gray)
        loss = self.criterion(output, color)
        loss.backward()

        self.optimizer.step()
        return loss.item()
    
    def eval_step(self, data):
        gray, color = data
        gray = gray.float().to(self.device)
        color = color.float().to(self.device)
        output = self(gray)
        loss = self.criterion(output, color)
        
        return loss.item()
        