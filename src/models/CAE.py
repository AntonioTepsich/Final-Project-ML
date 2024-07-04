import torch
import torch.nn as nn
from torch.optim import Adam



# Define the CAE architecture to colorize images (greyscale to LAB)

class CAE(nn.Module):
    def __init__(self, params):
        super(CAE, self).__init__()
        """
        encoder architecture explained:
        - input: 1x224x224
        - output: 8x28x28

        1. Conv2d: 1 input channel, 16 output channels, kernel size 3, stride 1, padding 1
        2. ReLU activation function
        3. MaxPool2d: kernel size 2, stride 2
        4. Conv2d: 16 input channels, 8 output channels, kernel size 3, stride 1, padding 1
        5. ReLU activation function
        6. MaxPool2d: kernel size 2, stride 2
        """


        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        decoder architecture explained:
        - input: 8x28x28
        - output: 2x224x224
        1. ConvTranspose2d: 8 input channels, 16 output channels, kernel size 3, stride 2, padding 1, output padding 1
        2. ReLU activation function
        3. ConvTranspose2d: 16 input channels, 2 output channels, kernel size 3, stride 2, padding 1, output padding 1
        4. Sigmoid activation function
        """
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


        self.device = params.device
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=params.learning_rate)

        
         
    def forward(self, x):
        z = self.encoder(x)
        x_re = self.decoder(z)
        return x_re
    
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
    

    def predict(self, condition):
        condition = condition.float().to(self.device)
        return self(condition)