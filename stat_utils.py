"""
Some tools to help with statistics.
"""

from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import pandas as pd

# set logging level to info, and set the format of the log to show the time
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class EpochResult:
    index: int  # epoch index
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_time: float  # in seconds


def log_one_epoch_stats(epoc_result: EpochResult):
    logging.info(
        f"Epoch {epoc_result.index + 1} \t train_loss: {epoc_result.train_loss:.3f} \t train_acc: {epoc_result.train_acc:.3f} \t test_loss: {epoc_result.test_loss:.3f} \t test_acc: {epoc_result.test_acc:.3f} \t epoch_time: {epoc_result.epoch_time:.3f} seconds"
    )


def write_csv_header(file_path):
    with open(file_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,epoch_time\n")


def write_one_epoch_stats_to_csv(epoc_result: EpochResult, file_path):
    with open(file_path, "a") as f:
        f.write(
            f"{epoc_result.index},{epoc_result.train_loss},{epoc_result.train_acc},{epoc_result.test_loss},{epoc_result.test_acc},{epoc_result.epoch_time}\n"
        )


def plot_loss(csv_file_path: str) -> None:
    """
    Plot the loss curve.
    """
    # use pandas df and convert the loss column to a list of floats
    df = pd.read_csv(csv_file_path)
    train_loss = df["train_loss"].tolist()
    test_loss = df["test_loss"].tolist()
    # plot the loss curve
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.legend()
    plt.show()


def plot_acc(csv_file_path: str) -> None:
    """
    Plot the accuracy curve.
    """
    # use pandas df and convert the loss column to a list of floats
    df = pd.read_csv(csv_file_path)
    train_acc = df["train_acc"].tolist()
    test_acc = df["test_acc"].tolist()
    # plot the loss curve
    plt.plot(train_acc, label="train")
    plt.plot(test_acc, label="test")
    plt.legend()
    plt.show()

def plot_error(csv_file_path: str) -> None:
    """
    Plot the error curve.
    """
    # use pandas df and convert the loss column to a list of floats
    df = pd.read_csv(csv_file_path)
    train_acc = df["train_acc"].tolist()
    test_acc = df["test_acc"].tolist()
    # plot the loss curve
    # plt.plot([1 - acc for acc in  train_acc], label="train")
    x_ticks = [10,100,1000]
    # scale
    plt.plot([1 - acc for acc in  test_acc], label="test")
    plt.xscale("log")
    plt.xticks(x_ticks, x_ticks)
    plt.legend()
    plt.show()


def plot_errors(zero_noise_csv: str, small_noise_csv: str, large_noise_csv: str, dataset: str) -> None:
    """
    Plot the test and train error curves for a given date and dataset.
    """
    df_zero = pd.read_csv(zero_noise_csv)
    df_small = pd.read_csv(small_noise_csv)
    df_large = pd.read_csv(large_noise_csv)

    test_acc_zero = df_zero["test_acc"].tolist()
    test_acc_small = df_small["test_acc"].tolist()
    test_acc_large = df_large["test_acc"].tolist()

    train_acc_zero = df_zero["train_acc"].tolist()
    train_acc_small = df_small["train_acc"].tolist()
    train_acc_large = df_large["train_acc"].tolist()


    # plot the loss curve
    plt.plot([1 - acc for acc in test_acc_zero], label="0% noise")
    plt.plot([1 - acc for acc in test_acc_small], label="10% noise")
    plt.plot([1 - acc for acc in test_acc_large], label="20% noise")

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    x_ticks = [10,100,1000]
    # scale
    plt.xscale("log")
    plt.xticks(x_ticks, x_ticks)
    plt.title(f"{dataset} Test Error Rate")
    plt.legend()
    plt.show()

    # plot the loss curve
    plt.plot([1 - acc for acc in train_acc_zero], label="0% noise")
    plt.plot([1 - acc for acc in train_acc_small], label="10% noise")
    plt.plot([1 - acc for acc in train_acc_large], label="20% noise")

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    x_ticks = [10,100,1000]
    # scale
    plt.xscale("log")
    plt.xticks(x_ticks, x_ticks)
    plt.title(f"{dataset} Train Error Rate")
    plt.legend()
    plt.show()