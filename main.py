from transformers import BertTokenizer
from Datasets.FIQA_SA import FIQA_SA
from Models.FinBERT_SA import FinBERT_SA
from torch.utils.data import Dataset

class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# Instance creation
dataset_loader = FIQA_SA(tokenizer)
# __call__ method will be called
train, test  = dataset_loader()

#convert dataset to torch.utils.data
train_ds = HFDataset(train)
test_ds = HFDataset(test)

#model instance and train
model = FinBERT_SA()
model.train(train_ds, test_ds)