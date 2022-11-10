import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset #HuggingFace

from config import Config
from Datasets.utils import getGdriveDataset, unzipFile

class FIQA_SA:
    def __init__(self, tokenizer):
        self.files = ["task1_post_ABSA_train.json", "task1_headline_ABSA_train.json"]
        self.path = os.path.join(os.getcwd(), "data", "FIQA_SA")
        os.makedirs(self.path, exist_ok=True)
        for _f in self.files:
            if not os.path.exists(os.path.join(self.path, _f)):
                zip_path = os.path.join(self.path, "FIQA_SA.zip")
                getGdriveDataset(Config()().get("Dataset").get("FIQA_SA"), zip_path)
                unzipFile(zip_path, self.path)
        self.processFiles()
        self.formDataset(tokenizer)

    def processFiles(self):
        self.data = pd.DataFrame(columns=["_id", "sentence", "aspects", "sentiment_score", "label"])
        data_dict = {}
        for _f in self.files:
            path = os.path.join(self.path, _f)
            current_data = json.loads(open(path, "r").read())
            data_dict.update(current_data)
        for _id, item_dict in data_dict.items():
            sentence = item_dict["sentence"]
            aspects = item_dict["info"][0]["aspects"][2:-2]
            sentiment_score = float(item_dict["info"][0]["sentiment_score"])
            if sentiment_score >= -1/3 and sentiment_score <= 1/3:
                # Neutral
                sentiment_label = 0
            elif sentiment_score > 1/3:
                # Positive
                sentiment_label = 1
            else:
                # Negative
                sentiment_label = 2
            self.data.loc[len(self.data)] = [_id, sentence, aspects, sentiment_score, sentiment_label]

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
        return self.dataset_train, self.dataset_val