#
# Created by Aman LaChapelle on 4/9/17.
#
# helmai
# Copyright (c) 2017 Aman LaChapelle
# Full license at helmai/LICENSE.txt
#

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
from tensorboard_logger import configure, log_value
import numpy as np
import matplotlib.pyplot as plt


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.model = models.resnet18()

        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


def calc_confusion_matrix(net, test_dataset):  # check this thing - not sure if it is quite right

    confusion_matrix = torch.zeros(10, 10)
    net.train(False)
    net.cuda()

    for i, data in enumerate(test_dataset, int(0.98 * len(test_dataset))):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        outputs = net(inputs)

        for j in range(labels.size()[0]):
            confusion_matrix[labels.data[j]] += outputs[j].squeeze().cpu().data

    net.train(True)

    for i in range(10):
        confusion_matrix[i] = softmax(Variable(confusion_matrix[i])).data  # normalize the confusion matrix rows

    error = np.diagonal(np.abs(np.eye(10) - confusion_matrix.cpu().numpy()))

    return confusion_matrix.numpy(), error


def validate(net, test_dataset, criterion):
    net.train(False)
    running_loss = 0.0
    for i, data in enumerate(test_dataset, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda(async=True)
        labels = Variable(labels).cuda(async=True)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]

    print("Total Loss: %.3f" % (running_loss/len(test_dataset)))


def train_until_plateau(net, train_dataset, test_dataset, criterion,
                        k=1000, starting_lr=1e-3, ending_lr=1e-6,
                        max_epochs=5, weight_decay=1e-5):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    current_lr = starting_lr
    optimizer = optim.Adam(net.parameters(), current_lr, weight_decay=weight_decay)
    accumulated_loss = []

    net.train()
    net.cuda()

    for epoch in range(max_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            inputs = Variable(inputs).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            if i % 25 == 24:  # at certain intervals register the backward hook on the gradient
                current_step = i + 1 + len(train_dataset) * epoch

                h1 = net.stem.register_backward_hook(
                    lambda _1, _2, grad: log_value("stem grad", torch.mean(grad[0]).data[0], step=current_step)
                )
                h2 = net.layer1.register_backward_hook(
                    lambda _1, _2, grad: log_value("layer1 grad", torch.mean(grad[0]).data[0], step=current_step)
                )
                h3 = net.layer2.register_backward_hook(
                    lambda _1, _2, grad: log_value("layer2 grad", torch.mean(grad[0]).data[0], step=current_step)
                )
                h4 = net.layer3.register_backward_hook(
                    lambda _1, _2, grad: log_value("layer3 grad", torch.mean(grad[0]).data[0], step=current_step)
                )
                h5 = net.layer4.register_backward_hook(
                    lambda _1, _2, grad: log_value("layer4 grad", torch.mean(grad[0]).data[0], step=current_step)
                )
                h6 = net.linear.register_backward_hook(
                    lambda _1, _2, grad: log_value("linear grad", torch.mean(grad[0]).data[0], step=current_step)
                )

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            accumulated_loss.append(loss.data[0])

            if i % 25 == 24:
                current_step = i + 1 + len(train_dataset) * epoch

                h1.remove()  # remove the hook so we only record the gradients every 10 steps (log_value is slow)
                h2.remove()
                h3.remove()
                h4.remove()
                h5.remove()
                h6.remove()

                log_value("Loss", loss.data[0], step=current_step)

            if i % k == k-1:
                print('[%d, %5d] avg loss: %.3f' % (epoch + 1, i + 1, running_loss / k))
                conf_mat, accuracy = calc_confusion_matrix(rnet, test_dataset)

                print('Accuracy per class:')
                for i in range(10):
                    print(classes[i], " ", "%.5f" % (1.0-accuracy[i]))
                    log_value(classes[i], (1.0-accuracy[i]), step=current_step)

                img = plt.matshow(conf_mat)
                plt.colorbar(img)
                plt.savefig("training/run_2_plots/%d" % (current_step))
                plt.close()

                # looks like 0.2*current_lr trains slower but better
                if np.abs(np.mean(np.diff(np.array(accumulated_loss)))) <= 0.5 * current_lr:

                    current_lr = np.max([current_lr * 1e-1, ending_lr])

                    log_value("LR", current_lr, step=current_step)
                    optimizer = optim.Adam(net.parameters(), current_lr, weight_decay=weight_decay)
                    accumulated_loss.clear()

                running_loss = 0.0


if __name__ == "__main__":
    configure("training/run_2")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_cifar10 = datasets.CIFAR10("data", train=True, transform=transform, download=True)
    test_cifar10 = datasets.CIFAR10("data", train=False, transform=transform, download=True)

    traindata = DataLoader(train_cifar10, batch_size=4, shuffle=True, num_workers=4)
    testdata = DataLoader(test_cifar10, batch_size=4, shuffle=True, num_workers=4)

    rnet = ResNet18()

    criterion = nn.CrossEntropyLoss()

    train_until_plateau(rnet, traindata, testdata, criterion)
    validate(rnet, testdata, criterion)

