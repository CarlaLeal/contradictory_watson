import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer

class EntailmentDataset(Dataset):
    def __init__(self, entailment_csv, max_length):
        self.dataset = pd.read_csv(entailment_csv)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        hypothesis = self.dataset.loc[idx, ["hypothesis"]].values[0]
        premise = self.dataset.loc[idx, ["premise"]].values[0]
        label = self.dataset.loc[idx, ["label"]].values[0]
        inputs = self.tokenizer.encode_plus(
            hypothesis,
            premise,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor([label], dtype=torch.float) 
        }


