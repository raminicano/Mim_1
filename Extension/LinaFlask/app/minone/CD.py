import torch
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, data, tokenizers, mode = "train"):
        super().__init__()
        self.text = [str(i) for i in data['req_content'].values]
        self.labels = data['topic']
        # self.regions = data['gu_name']
        self.tokenizer = tokenizers
        self.mode = mode

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.text[idx], padding = 'max_length', max_length = 512, truncation = True, return_tensors = 'pt')

        inputs_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        label = self.labels[idx]

        if self.mode == "train":
            return inputs_ids, attention_mask, label
        else:
            return inputs_ids, attention_mask

class FeatureExtractDataset(Dataset):
    def __init__(self, data, tokenizers):
        super().__init__()
        # self.text = [str(i) for i in data['clean'].values]
        self.text = [str(i) for i in data]
        self.tokenizer = tokenizers

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.text[idx], padding = 'max_length', max_length = 512, truncation = True, return_tensors = 'pt')
        inputs_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return inputs_ids, attention_mask