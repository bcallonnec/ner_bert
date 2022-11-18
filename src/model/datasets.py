"""
Define Dataset objects used by pytorch models.
"""
from typing import Any, Dict, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset


class NERDataset(Dataset):
    """Dataset designed for NER task"""

    def __init__(
        self,
        texts: Union[str, Sequence[Any]],
        tokenizer: Any,
        max_len: int,
        labels: Sequence[Any],
        loss_ignore_index: int = -100,
        propagate_label_to_word_pieces: bool = False,
    ) -> None:
        """
        Init function
        Parameters
        ----------
        texts: Union[str, Sequence[Any]]
            List of tokenized text. The sentence is tokenized into a list of token.
        tokenizer: Any
            Usually a pretrained tokenizer from HuggingFace
        max_len: int
            the max len of the list of tokens
        labels: Sequence[Any]
            The corresponding tag of each token
        loss_ignore_index: int
            Label index that will be ignore by the loss function
        propagate_label_to_word_pieces: bool
            Wether to propagate the label of the word to all of its word pieces or not
        """
        # Convert str sentence to list of tokens
        texts = [elem.split() if isinstance(elem, str) else elem for elem in texts]

        # Set class attributes
        self.texts: Sequence[Any] = texts
        self.labels: Sequence[Any] = labels
        self.tokenizer: Any = tokenizer
        self.max_len: int = max_len
        self.loss_ignore_index: int = loss_ignore_index
        self.propagate_label_to_word_pieces: bool = propagate_label_to_word_pieces

    def __len__(self) -> int:
        """Get len of dataset"""
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Sequence]:
        """Get item at index"""
        text = self.texts[index]
        tags = self.labels[index]

        ids = []
        target_tag = []

        for idx, token in enumerate(text):
            inputs = self.tokenizer(
                token,
                truncation=True,
                max_length=self.max_len,
            )
            # remove special tokens <s> and </s>
            ids_ = inputs["input_ids"][1:-1]

            input_len = len(ids_)
            ids.extend(ids_)

            if self.propagate_label_to_word_pieces:
                target_tag.extend([tags[idx]] * input_len)
            else:
                target_tag.append(tags[idx])
                target_tag.extend([self.loss_ignore_index] * (input_len - 1))

        ids = ids[: self.max_len - 2]
        target_tag = target_tag[: self.max_len - 2]

        # Reconstruct specials tokens at start and end
        ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        target_tag = [self.loss_ignore_index] + target_tag + [self.loss_ignore_index]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([self.loss_ignore_index] * padding_len)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(target_tag, dtype=torch.long),
        }

    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Get data loader from dataset"""
        data_loader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        return DataLoader(self, **data_loader_params)
