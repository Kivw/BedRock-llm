import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

__all__ = ["PretrainDataset", "PretrainDataCollator"]


class PretrainDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset

    def tokenize_function(self, sample):
        outputs = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        return outputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.tokenize_function(self.dataset[index])

class PretrainDataCollator(DataCollatorMixin):
    def __init__(
        self,
        pad_token_id: int | None = 0,
        end_token_id: int | None = 2,
        bos_token_id: int | None = 1,
        ignore_index: int = -100,
        return_tensors: str = "pt"
    ):
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.bos_token_id = bos_token_id
        self.ignore_index = ignore_index
        self.return_tensors = return_tensors

    def torch_call(self, examples: list[dict[str, any]]):
        input_ids_list = []

        # 1️⃣ 拼接 BOS/EOS
        for ex in examples:
            ids = ex["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)  # 转成 tensor
            ids = torch.cat([
                torch.tensor([self.bos_token_id], dtype=torch.long),
                ids,
                torch.tensor([self.end_token_id], dtype=torch.long)
            ])
            input_ids_list.append(ids)

        # 2️⃣ 找到 batch 内最大长度
        max_len = max(len(ids) for ids in input_ids_list)

        batch_input_ids = []
        batch_attention_mask = []

        # 3️⃣ pad 
        for ids in input_ids_list:
            seq_len = len(ids)
            pad_len = max_len - seq_len

            # input_ids pad
            padded_input_ids = torch.cat([
                ids,
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])
            batch_input_ids.append(padded_input_ids)

            # attention mask
            attention_mask = (padded_input_ids != self.pad_token_id).long()
            batch_attention_mask.append(attention_mask)

        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_input_ids), # transformers中的loss_fn会自动对labels进行shift
            "attention_mask": torch.stack(batch_attention_mask),
        }

        return batch
