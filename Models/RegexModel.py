import os
import re
import nltk
from nltk import tokenize
from nltk.corpus import stopwords

from config import Config


class RegexModel:

    def __init__(self):
        self.regexForFigures = r"[-+]?(?:\d*\.\d+|\d+)"
        nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
        self.stopwords = stopwords.words("english")

    def filterTokens(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token.lower() not in self.stopwords:
                filtered_tokens.append(token.lower())
        return filtered_tokens

    def jaccardSimilarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection)/union
        
    def getAnswer(self, question, context):
        allSentences = tokenize.sent_tokenize(context)
        filtered_question_tokens = self.filterTokens(tokenize.word_tokenize(question))
        relevantSentences = []
        for sentence in allSentences:
            filtered_sentence_tokens = self.filterTokens(tokenize.word_tokenize(sentence))
            similarity = self.jaccardSimilarity(filtered_question_tokens, filtered_sentence_tokens)
            relevantSentences.append((sentence, similarity))
        relevantSentences.sort(key=lambda x: x[1], reverse=True)
        allFigures = re.findall(self.regexForFigures, relevantSentences[0][0])
        return relevantSentences[0][0], allFigures
