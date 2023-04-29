import torchvision.datasets as datasets
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Define the path where you want to download the data
data_path = "./data"

# Download the Food101 dataset
train_data = datasets.Food101(root=f"{data_path}/food101", download=True)
