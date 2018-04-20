import torch
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm


def get_outputs(model, loader):
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), leave=False):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs, volatile=True))
        outputs_full.append(outputs)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    labels_full = Variable(labels_full, volatile=True)
    return outputs_full, labels_full


def argmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return accuracy


def calc_accuracy(model, loader):
    outputs, labels = get_outputs(model, loader)
    accuracy = argmax_accuracy(outputs, labels)
    return accuracy
