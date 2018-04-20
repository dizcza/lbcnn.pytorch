import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from utils import calc_accuracy

from lbcnn_model import Lbcnn


class Trainer(object):
    def __init__(self, lbcnn_depth, batch_size):
        self.batch_size = batch_size
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epoch_count = 50
        self.lbcnn_depth = lbcnn_depth
        self.lr_scheduler_step = 5
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=4)

        model = Lbcnn(depth=self.lbcnn_depth).cuda()
        best_accuracy = 0.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=self.learning_rate,
                              momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_step)

        for epoch in range(self.epoch_count):
            scheduler.step()
            for batch_id, (inputs, labels) in enumerate(
                    tqdm(trainloader, desc="Epoch {}/{}".format(epoch, self.epoch_count))):
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            epoch_accuracy = calc_accuracy(model, loader=trainloader)
            print("Epoch {} accuracy: {}".format(epoch, epoch_accuracy))
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(model.state_dict(), os.path.join(self.models_dir, 'lbcnn_best.pt'))
        print('Finished Training')


if __name__ == '__main__':
    Trainer(lbcnn_depth=1, batch_size=16).train()
