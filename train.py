import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
import torch.nn as nn

from lbcnn_model import Lbcnn


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)

    net = Lbcnn()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda param: param.requires_grad, net.parameters()), lr=1e-4, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    for epoch in range(50):
        scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 200 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f  lr=%f' %
                      (epoch + 1, i + 1, running_loss / 2000, scheduler.get_lr()[0]))
                running_loss = 0.0
    torch.save(net, 'lbcnn.pt')
    print('Finished Training')


if __name__ == '__main__':
    train()
