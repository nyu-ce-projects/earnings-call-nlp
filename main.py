from transformers import BertTokenizer
from Datasets.FIQA_SA import FIQA_SA
from Datasets.FIQA_Aspect import FIQA_Aspect
from Models.FinBERT_SA import FinBERT_SA
from Models.FinBERT_Aspect import FinBERT_Aspect
from Datasets.HFDataset import HFDataset

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')

"""## Sentiment Analysis"""

# Instance created
fiqa_sa_data = FIQA_SA(tokenizer)
# __call__ method will be called
train_fiqa_sa, test_fiqa_sa  = fiqa_sa_data()

train_fiqa_sa, test_fiqa_sa = HFDataset(train_fiqa_sa), HFDataset(test_fiqa_sa)

model = FinBERT_SA()

model.train(train_fiqa_sa, test_fiqa_sa)

"""## Aspect Detection"""

# Instance created
fiqa_ad_data = FIQA_SA(tokenizer)
# __call__ method will be called
train_fiqa_ad, test_fiqa_ad  = fiqa_ad_data()

#Define model
model_ad = FinBERT_Aspect()

model_ad.train(train_fiqa_ad, test_fiqa_ad)