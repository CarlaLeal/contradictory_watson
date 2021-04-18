from torch import nn
from transformers import RobertaForMultipleChoice

class RoBERTaNLI(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification = RobertaForMultipleChoice.from_pretrained('roberta-base')

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.classification(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
