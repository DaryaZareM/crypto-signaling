import transformers
import torch
class BERTweetclass(torch.nn.Module):
    def __init__(self,model_path):
        super(BERTweetclass, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained(model_path,
                                                         return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output