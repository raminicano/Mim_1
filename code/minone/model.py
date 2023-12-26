import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, model, in_features, num_label):
        super(CustomModel, self).__init__()
        self.model = model
        self.in_features = in_features
        self.num_label = num_label

        self.pooler = nn.Linear(in_features=in_features, out_features=in_features)
        self.tanh = nn.Tanh()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_label)
        )

    def forward(self, input_ids, mask):
        outputs = self.model(input_ids=input_ids, attention_mask=mask)

        last_hidden_output = outputs.last_hidden_state
        cls_output = last_hidden_output[:, 0, :]

        pooled_output = self.tanh(self.pooler(cls_output))
        pred = self.classifier(pooled_output)

        return cls_output, pred