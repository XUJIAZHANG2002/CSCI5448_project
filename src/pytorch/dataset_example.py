from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def __len__(self):
        return len(self.data)

# Create dataset and loader
dataset = MyDataset(torch.randn(1000, 32), torch.randint(0, 10, (1000,)))
sampler = RandomSampler(dataset)
loader = DataLoader(dataset, batch_size=32, sampler = sampler,shuffle=False, num_workers=2)


for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
