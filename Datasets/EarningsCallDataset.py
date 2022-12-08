import os


class EarningsCallDataset:

    def __init__(self):
        self.path = os.path.join(os.getcwd(), "data", "EarningsCall", "ACL19_Release")
        self.dataset = {}
        for directory in os.listdir(self.path):
            companyName, date = directory.split("_")
            transcript = open(os.path.join(self.path, directory, "TextSequence.txt"), "r").read()
            if companyName not in self.dataset:
                self.dataset[companyName] = [[date, transcript]]
            else:
                self.dataset[companyName].append([date, transcript])

if __name__ == "__main__":
    EarningsCallDataset()
