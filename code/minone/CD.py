import torch
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, data, tokenizers, mode="train"):
        """
        ClassifierDataset의 생성자

        Args:
            data (DataFrame): 입력 데이터 프레임.
            tokenizers: 텍스트를 토큰화하기 위한 토크나이저.
            mode (str): 데이터셋 모드. "train"인 경우 레이블(label)이 포함되어 있고, 그렇지 않은 경우 "inference" 또는 다른 것으로 설정됩니다.
        """
        super().__init__()
        self.text = [str(i) for i in data['req_content'].values]
        self.labels = data['topic']
        self.tokenizer = tokenizers
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터를 반환합니다.

        Args:
            idx (int): 데이터 인덱스.

        Returns:
            torch.Tensor: 입력 IDs.
            torch.Tensor: 어텐션 마스크.
            int: 레이블 (train 모드인 경우).
        """
        inputs = self.tokenizer(self.text[idx], padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        inputs_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        label = self.labels[idx]

        if self.mode == "train":
            return inputs_ids, attention_mask, label
        else:
            return inputs_ids, attention_mask

class FeatureExtractDataset(Dataset):
    def __init__(self, data, tokenizers):
        """
        FeatureExtractDataset의 생성자

        Args:
            data (list): 입력 데이터 리스트.
            tokenizers: 텍스트를 토큰화하기 위한 토크나이저.
        """
        super().__init__()
        self.text = [str(i) for i in data]
        self.tokenizer = tokenizers

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터를 반환합니다.

        Args:
            idx (int): 데이터 인덱스.

        Returns:
            torch.Tensor: 입력 IDs.
            torch.Tensor: 어텐션 마스크.
        """
        inputs = self.tokenizer(self.text[idx], padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        inputs_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return inputs_ids, attention_mask
