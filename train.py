from tqdm import tqdm
import copy
import atexit
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn

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
        self.best_model = None
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        atexit.register(self.save_best_model)

    def save_best_model(self):
        if self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.models_dir, 'lbcnn_best.pt'))

    def train(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)

        model = Lbcnn(depth=self.lbcnn_depth).cuda()
        best_accuracy = 0.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=self.learning_rate,
                              momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_step)
        print_step = len(trainloader) // 10

        for epoch in range(self.epoch_count):
            scheduler.step()
            running_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            for i, data in enumerate(tqdm(trainloader, desc="Epoch {}".format(epoch))):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                optimizer.zero_grad()
                predicted_proba = model(inputs)
                loss = criterion(predicted_proba, labels)
                predicted_labels = torch.max(predicted_proba.data, dim=1)[1]
                total_samples += len(labels)
                epoch_correct += (predicted_labels == labels.data).sum()

                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if i % print_step == 0:
                    print('[{}, {:5d}] loss={:.3f}  lr={:.1e}'.format(epoch, i, running_loss, scheduler.get_lr()[0]))
                    running_loss = 0.0
            epoch_accuracy = epoch_correct / float(total_samples)
            torch.save({
                'epoch': epoch,
                'lbcnn.depth': self.lbcnn_depth,
                'best_accuracy': best_accuracy,
                'model.state_dict': model.state_dict(),
                'optimizer.state_dict': optimizer.state_dict(),
            }, os.path.join(self.models_dir, 'checkpoint.pt'))
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                self.best_model = copy.deepcopy(model)
        print('Finished Training')


if __name__ == '__main__':
    Trainer(lbcnn_depth=20, batch_size=16).train()
