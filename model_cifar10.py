import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--noise", type=float, default=0.0)

args = parser.parse_args()


from stat_utils import (
    EpochResult,
    log_one_epoch_stats,
    write_csv_header,
    write_one_epoch_stats_to_csv,
)

data_path = "./data"
NUM_EPOCHS = 2000
current_date = time.strftime("%Y-%m-%d")
epoch_stats_csv = f"stats/epoch_stats_cifar10_{current_date}_noise_{args.noise*100}.csv"
# set manual_seed
torch.manual_seed(42)

# Define the transformation for the images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

train_data = torchvision.datasets.CIFAR10(root='./cifar100', train=True, transform=transform)

# calculate number of labels to change
num_labels = int(args.noise * len(train_data))

# create list of indices to change
random.seed(42)
indices_to_change = random.sample(range(len(train_data)), num_labels)

# change labels
for i in indices_to_change:
    label = train_data.targets[i]
    new_label = random.randint(0, 9)
    while new_label == label:
        new_label = random.randint(0, 9)
    train_data.targets[i] = new_label


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, )

test_data = torchvision.datasets.CIFAR10(root='./cifar100', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=True,)

print(f"train_set has {len(train_data)} images")
print(f"test_set has {len(test_data)} images")

# Define the ResNet18 model
model = torchvision.models.resnet18(weights=None)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 10
                     )  # 101 is the number of classes in Food101 dataset

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"model will be trained on device={device}")
model = model.to(device)
train_loss_history = []
test_loss_history = []
test_acc_history = []

write_csv_header(epoch_stats_csv)

# use tqdm to show a progress bar
# for epoch in tqdm(range(NUM_EPOCHS)):
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    # food101 the model, save the food101 loss and test loss on the entire epoch to the history
    train_loss = 0.0
    test_loss = 0.0
    model.train()  # set the model to training mode
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # get the training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = float(train_correct) / train_total
    train_loss_history.append(train_loss / len(train_loader))

    # test the model, save the food101 loss and test loss on the entire epoch to the history
    model.eval()  # set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        # for i, data in enumerate(tqdm(test_loader)):
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # get the test accuracy
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        test_loss_history.append(test_loss / len(test_loader))
        test_acc_history.append(correct / total)

    # after the epoch is done, log the stats, and write them to a csv file.
    epoch_result = EpochResult(
        epoch,
        train_loss_history[-1],
        train_acc,
        test_loss_history[-1],
        test_acc_history[-1],
        time.time() - start_time,
    )
    log_one_epoch_stats(epoch_result)
    write_one_epoch_stats_to_csv(epoch_result, epoch_stats_csv)

    if epoch % 250 == 0:
        # save the model every 50 epochs
        torch.save(model.state_dict(), f"./models/food101_resnet18_{epoch}.pth")

# Test the model after training
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")