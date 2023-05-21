"""
Depreciated.
use model_food101.py instead.

"""





import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import random
import argparse
from stat_utils import (
    EpochResult,
    log_one_epoch_stats,
    write_csv_header,
    write_one_epoch_stats_to_csv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.1)

    args = parser.parse_args()

    data_path = "./data"
    NUM_EPOCHS = 1051
    current_date = time.strftime("%Y-%m-%d")
    epoch_stats_csv = f"stats/epoch_stats_food101_{current_date}_noise_{args.noise*100}.csv"
    # set manual_seed
    torch.manual_seed(42)

    # Define the transformation for the image
    # taken from https://github.com/shubhajitml/food-101/blob/master/food-101-pytorch.ipynb

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


    # Load the datasets
    train_data = datasets.Food101(
        root=f"{data_path}/food101", split="train", transform=train_transform
    )

    # create list of indices to change
    num_labels = int(args.noise * len(train_data))
    random.seed(42)
    indices_to_change = random.sample(range(len(train_data)), num_labels)

    # change labels
    for i in indices_to_change:
        label = train_data._labels[i]
        new_label = random.randint(0, 100)
        while new_label == label:
            new_label = random.randint(0, 100)
        train_data._labels[i] = new_label

    test_data = datasets.Food101(
        root=f"{data_path}/food101", split="test", transform=test_transform
    )
    # train_set = ImageFolder('./data/food101', transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=3)
    # test_set = ImageFolder('./data/test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=3)

    print(f"train_set has {len(train_data)} images")
    print(f"test_set has {len(test_data)} images")

    # Define the ResNet18 model
    model = torchvision.models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 101)  # 101 is the number of classes in Food101 dataset

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"model will be trained on device={device}")
    print(f"noisiness={args.noise}")

    #print the device name
    print(torch.cuda.get_device_name(0))
    model.to(device)
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
        # start_time2 = time.time()
        for i, data in enumerate(train_loader):
            # if i % 50 == 0:
            #     print(f"batch {i} time: {time.time() - start_time2}")
            #     start_time2 = time.time()
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
            for i, data in enumerate(tqdm(test_loader)):
                # for i, data in enumerate(test_loader):
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

        if epoch % 50 == 0:
            # save the model every 50 epochs
            torch.save(model.state_dict(), f"./models/food101_resnet18_{epoch}_noise_{args.noise*100}_{current_date}.pth")

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