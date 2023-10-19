import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')


        num_features = self.resnet.fc.in_features 
        self.resnet.fc = nn.Linear(num_features , num_classes)

        

    def forward(self, x):
        

        x = self.resnet(x)
        return x



if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)


    args = ARGS(
        epochs= 50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=0.00005,  #TODO,
        batch_size= 32,#TODO,
        step_size= 15, #TODO,
        gamma= 0.1#TODO
    )

    
    print(args)


    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
