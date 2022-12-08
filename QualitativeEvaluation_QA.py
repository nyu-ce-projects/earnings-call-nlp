from Models.RoBERTa_QA import RoBERTa_QA
from Models.RegexModel import RegexModel
from Datasets.EarningsCallDataset import EarningsCallDataset


dataset = EarningsCallDataset()
model1 = RoBERTa_QA()
model2 = RegexModel()

companies = [
            "Adobe Systems Inc",
            "Twitter, Inc.",
            "Nike",
            "McDonald's Corp."
            ]
questions = [
            "What was the revenue this quarter?", 
            "What was the net income this quarter?", 
            "What are the Earnings Per Share or EPS in this quarter?", 
            "What are the Earnings before taxes or EBIT this quarter?",
            "What are the projected earnings for next quarter?"] 

for company in companies:
    companyDataset = dataset.dataset[company]
    date = companyDataset[0][0]
    context = companyDataset[0][1]
    print("*"*50)
    print("Selected Company : {}, Date: {}".format(company, date))
    for question in questions:
        print("="*25)
        print(question)
        print("="*5 + " RoBERTa_QA Answer " + "="*5)
        print(model1.getAnswer(question, context))
        print("="*5 + " RegexModel Answer " + "="*5)
        print(model2.getAnswer(question, context))
        print("="*25)
    print("*"*50)
