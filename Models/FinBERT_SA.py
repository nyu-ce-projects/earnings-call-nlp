import os
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from config import Config
from metrics import compute_metrics


class FinBERT_SA:

    def __init__(self, pretrained_path=""):
        if pretrained_path == "":
            pretrained_path = 'yiyanghkust/finbert-tone'
        self.finbert = BertForSequenceClassification.from_pretrained(pretrained_path, num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.pipeline = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)
        self.trained_models_path = os.path.join(os.getcwd(), "finetuned_models", "sentiment_analysis")
        self.label = []
        self.preds = []

    def get_precision_recall(self):
        precision, recall, fscore, _ = precision_recall_fscore_support(self.label, self.preds, average='weighted')
        return precision, recall, fscore

    def compute(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        self.label += [i for i in labels]
        self.preds += [i for i in predictions]
        return {'accuracy' : accuracy_score(predictions, labels)}

    def train(self, train_dataset, val_dataset):
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
                metric_for_best_model = 'accuracy',
                logging_strategy = 'epoch'
        )
        self.trainer = Trainer(
                    model = self.finbert,
                    args = args,
                    train_dataset = train_dataset,
                    eval_dataset = val_dataset,
                    compute_metrics = self.compute #compute_metrics
        )
        self.trainer.train()
        self.pipeline = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)
        return self.trainer

    def checkSentiment(self, sentence):
        return self.pipeline(sentence)
