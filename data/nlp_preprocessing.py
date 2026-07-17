import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = []
        for text in positive + negative:
            vocab.extend(text.split(" "))
        vocab = sorted(list(set(vocab)))
        
        vocab_ids = {word: idx + 1 for idx, word in enumerate(vocab)}

        tensors = []
        for text in (positive + negative):
            # Convert this individual sentence directly into a 1D PyTorch float tensor
            seq_ids = [float(vocab_ids[word]) for word in text.split(" ")]
            tensors.append(torch.tensor(seq_ids))
        res = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
        return res
