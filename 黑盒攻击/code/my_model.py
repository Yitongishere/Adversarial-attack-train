import torch
from torch import nn

class CNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.layers = nn.Sequential(    

                nn.Conv2d(1, 25, 3, 1, 2),
                nn.BatchNorm2d(25),       
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(25, 50, 3, 1, 2), 
                nn.BatchNorm2d(50),    
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten(),

                nn.Linear(8 * 8 * 50, 200),
                nn.ReLU(),

                nn.Linear(200, 10)
            )

    def forward(self, x):
        
        x = x.reshape((-1, 1, 28, 28))
        return self.layers(x)



        

if __name__ == "__main__":
    
    from fmnist_dataset import load_fashion_mnist
    from torch.utils.data import DataLoader

    train, dev, test = load_fashion_mnist("../../data")
    train_dataloader = DataLoader(train, batch_size=1)
    
    m = CNN()
    
    for x, y in train_dataloader:
        
        l = m(x)                # torch.Size([1, 10])
        break