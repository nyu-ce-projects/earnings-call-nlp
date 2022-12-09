import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset #HuggingFace

from config import Config
from Datasets.utils import getGdriveDataset, unzipFile
from Datasets.HFDataset import HFDataset

class FIQA_QA:
    def __init__(self, tokenizer):
        files = ["FiQA_train_doc_final.tsv", "FiQA_train_question_final.tsv", "FiQA_train_question_doc_final.tsv"]
        self.path = os.path.join(os.getcwd(), "data", "FIQA_QA")
        os.makedirs(self.path, exist_ok=True)
        for _f in files:
            if not os.path.exists(os.path.join(self.path, _f)):
                zip_path = os.path.join(self.path, "FIQA_QA.zip")
                getGdriveDataset(Config()().get("Dataset").get("FIQA_QA"), zip_path)
                unzipFile(zip_path, self.path)
        self.processFiles()
        self.formDataset(tokenizer)

    def processFiles(self):
        doc_path = os.path.join(self.path, "FiQA_train_doc_final.tsv")
        documents = pd.read_csv(doc_path, sep="\t", usecols=["docid", "doc", "timestamp"])
        questions_path = os.path.join(self.path, "FiQA_train_question_final.tsv")
        questions = pd.read_csv(questions_path, sep="\t", usecols=["qid", "question", "timestamp"])
        links_path = os.path.join(self.path, "FiQA_train_question_doc_final.tsv")
        links = pd.read_csv(links_path, sep="\t", usecols=["qid", "docid"])
        self.data = links.merge(questions, how="left", on='qid').drop("timestamp", axis=1)\
                         .merge(documents, how='left', on='docid').drop("timestamp", axis=1)
        self.data.dropna(inplace=True, axis=0)
        return self.data

    def formDataset(self, tokenizer):
        df_train, df_val = train_test_split(self.data, test_size=0.1, random_state=1234)
        self.dataset_train = Dataset.from_pandas(df_train)
        self.dataset_val = Dataset.from_pandas(df_val)
        #question_max_length = int(self.data["question"].apply(lambda x: len(str(x).split(" "))).quantile(0.9999))
        #doc_max_length = int(self.data["doc"].apply(lambda x: len(str(x).split(" "))).quantile(0.99))
        #max_len = question_max_length + doc_max_length
        max_len = 128
        self.dataset_train = self.dataset_train.map(lambda x: tokenizer(x['question'], x['doc'], \
                                                    truncation=True, padding='max_length', \
                                                    max_length=max_len),\
                                                    batched=True)
        self.dataset_val = self.dataset_val.map(lambda x: tokenizer(x['question'], x['doc'], \
                                                truncation=True, padding='max_length', \
                                                max_length=max_len), \
                                                batched=True)
        self.dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', \
                                                             'attention_mask', 'label'])
        self.dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', \
                                                           'attention_mask', 'label'])

    def __call__(self):
        return self.dataset_train, self.dataset_val