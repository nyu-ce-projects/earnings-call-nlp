import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset #HuggingFace
from torch.utils.data import Dataset

from config import Config
from Datasets.utils import getGdriveDataset, unzipFile

from FIQA_SA import FIQA_SA

#for conversion to torch.utils.data.Dataset
class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset
        
    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)

class FIQA_Aspect(FIQA_SA):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def process(self):
        self.data = pd.DataFrame(columns=["_id", "sentence", "aspects", "label"])
        data_dict = {}
        for _f in self.files:
            path = os.path.join(self.path, _f)
            current_data = json.loads(open(path, "r").read())
            data_dict.update(current_data)
        for _id, item_dict in data_dict.items():
            sentence = item_dict["sentence"]
            aspects = item_dict["info"][0]["aspects"][2:-2]
            aspects_list = aspects.split('/')
            if aspects_list[0] == 'Stock' and aspects_list[1] == 'Price Action':
                # stock/price action
                aspects_label = 0
            elif aspects_list[0] == 'Stock':
                # stock/others
                aspects_label = 1
            elif aspects_list[0] == 'Corporate':
                # corporate
                aspects_label = 2
            self.data.loc[len(self.data)] = [_id, sentence, aspects, aspects_label] 

    def formDataset(self, tokenizer):
        df_train, df_val = train_test_split(self.data, stratify=self.data['label'], test_size=0.1, random_state=1234)
        self.dataset_train = Dataset.from_pandas(df_train)
        self.dataset_val = Dataset.from_pandas(df_val)
        self.dataset_train = self.dataset_train.map(lambda x: tokenizer(x['sentence'], truncation=True,
                                                                padding='max_length', max_length=128), batched=True)
        self.dataset_val = self.dataset_val.map(lambda x: tokenizer(x['sentence'], truncation=True,
                                                                padding='max_length', max_length=128), batched=True)
        self.dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        self.dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])       
    
    def __call__(self):
        self.train_ds = HFDataset(self.dataset_train)
        self.test_ds = HFDataset(self.dataset_val)
        return self.train_ds, self.test_ds