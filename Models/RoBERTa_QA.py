import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import Trainer, TrainingArguments

from config import Config


class RoBERTa_QA:

    def __init__(self, pretrained_path=""):
        if pretrained_path == "":
            pretrained_path = 'deepset/roberta-base-squad2'
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        self.label = []
        self.preds = []

    def train(self, train_dataset=None, val_dataset=None):
        raise NotImplementedError

    def getAnswer(self, question, context):
        qa_input = {"question": question, "context": context}
        return self.pipeline(qa_input)