import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer_, max_len):
        self.tokenizer = tokenizer_
        self.data = dataframe
        self.tweet_normalize_ = dataframe.tweet_normalize
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet_normalize_)

    def __getitem__(self, index):

        tweet_normalize = str(self.tweet_normalize_[index])
        tweet_normalize = " ".join(tweet_normalize.split())
        inputs = self.tokenizer.encode_plus(
            tweet_normalize,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }