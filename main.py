import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm

from lbcnn_model import Lbcnn
from utils import calc_accuracy, get_mnist_loader

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lbcnn_best.pt')


def test(model=None):
    if model is None:
        assert os.path.exists(MODEL_PATH), "Train a model first"
        lbcnn_depth, state_dict = torch.load(MODEL_PATH)
        model = Lbcnn(depth=lbcnn_depth)
        model.load_state_dict(state_dict)
    loader = get_mnist_loader(train=False)
    accuracy = calc_accuracy(model, loader=loader, verbose=True)
    print("MNIST test accuracy: {:.3f}".format(accuracy))


def train(n_epochs=50, lbcnn_depth=2, learning_rate=1e-2, momentum=0.9, weight_decay=1e-4, lr_scheduler_step=5):
    start = time.time()
    models_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    train_loader = get_mnist_loader()
    model = Lbcnn(depth=lbcnn_depth)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    best_accuracy = 0.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay, nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step)

    for epoch in range(n_epochs):
        scheduler.step()
        for batch_id, (inputs, labels) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, n_epochs))):
            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_accuracy = calc_accuracy(model, loader=train_loader)
        print("Epoch {} train accuracy: {:.3f}".format(epoch, epoch_accuracy))
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save((lbcnn_depth, model.state_dict()), MODEL_PATH)
    train_duration_sec = int(time.time() - start)
    print('Finished Training. Total training time: {} sec'.format(train_duration_sec))


if __name__ == '__main__':
    train(n_epochs=5)
    test()
