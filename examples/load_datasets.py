from torch.utils.data import Dataset, DataLoader

from bayess.data import load_dataset

import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

train_data, train_labels, test_data, test_labels = load_dataset("mnist", flatten=False, binarize=True, with_targets=True)


use_cuda = False
batch_size = 512

class MyDataset(Dataset):
    def __init__(self, x, y):
        # super(MyDataset, self).__init__()
        self.x = torch.tensor(x)#.reshape(len(x), -1)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
train_dataset = MyDataset(train_data, train_labels)
test_dataset = MyDataset(test_data, test_labels)

if use_cuda:
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)
else:
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                            )
    valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            )