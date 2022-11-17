from torch.utils.data import Dataset

class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)