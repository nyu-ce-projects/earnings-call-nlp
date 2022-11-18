import os
import numpy as np
import pandas as pd 
import torch
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, pipeline
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from config import Config


class FinBERT_Aspect:

    def __init__(self):
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain',num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
        self.pipeline = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)
        self.trained_models_path = os.path.join(os.getcwd(), "finetuned_models", "aspect_detection")
        self.label = []
        self.preds = []

    def get_precision_recall(self):
        precision, recall, fscore, _ = precision_recall_fscore_support(self.label, self.preds, average='weighted')
        return precision, recall, fscore

    def compute(self,eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            self.label += [i for i in labels]
            self.preds += [i for i in predictions]
            return {'accuracy' : accuracy_score(predictions, labels)}

    def train(self, train_dataset, val_dataset, test_dataset=None):
        if not (isinstance(train_dataset, Dataset) and isinstance(val_dataset, Dataset)):
            raise TypeError("'train_dataset' and 'val_dataset' should be of type torch.utils.data.Dataset")
        args = TrainingArguments(
                output_dir = self.trained_models_path,
                evaluation_strategy = 'epoch',
                save_strategy = 'epoch',
                learning_rate = Config()().get("learning_rate", 2e-5),
                per_device_train_batch_size = Config()().get("batch_size", 32),
                per_device_eval_batch_size = Config()().get("batch_size", 32),
                num_train_epochs = Config()().get("num_epochs", 5),
                weight_decay = Config()().get("weight_decay", 0.01),
                load_best_model_at_end = True,
                metric_for_best_model = 'eval_loss',
        )
        trainer = Trainer(
                    model = self.finbert,
                    args = args,
                    train_dataset = train_dataset,
                    eval_dataset = val_dataset,
                    
        )
        trainer.compute_metrics = self.compute
        trainer.train()
        self.pipeline = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)

        if(test_dataset):
            self.finbert.eval()
            trainer.predict(test_dataset).metrics

    def detect_aspect(self, sentence):
        return self.pipeline(sentence)