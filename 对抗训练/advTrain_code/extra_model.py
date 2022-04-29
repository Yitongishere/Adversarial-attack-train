import torch
from torch import nn

class extraCNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(1, 40, 3, 1, 2),
                nn.ReLU(),
                nn.Conv2d(40, 60, 3, 2, 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(60, 80, 5, 3, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(6 * 6 * 80, 200),
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
    
    m = extraCNN()
    
    for x, y in train_dataloader:
        
        l = m(x)
        break