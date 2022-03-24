import torch
from models.model import MyCNN
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader

import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "data", "test")
weight_path = os.path.join(base_path, "weights", "weight.pth")

# load model and use weights we saved before
model = ExampleCNN()
# model = MyCNN()
model.load_state_dict(torch.load(weight_path))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path)

predict_correct = 0
model.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Testing"):
        data, target = data.to(device), target.to(device)

        output = model(data)
        predict_correct += (output.data.max(1)[1] == target.data).sum()
        
    accuracy = 100. * predict_correct / len(test_loader.dataset)
print(f'Test accuracy: {accuracy:.4f}%')