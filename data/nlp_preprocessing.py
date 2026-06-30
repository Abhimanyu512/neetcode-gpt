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
        tokenMap = defaultdict(int)
        uniqueWords = []
        pos_encoded, neg_encoded = [], []
        i=0
        for sentence in positive + negative:
            for word in sentence.split(' '):
                if word not in uniqueWords:
                    uniqueWords.append(word)
        uniqueWords = sorted(uniqueWords)
        for word in uniqueWords:
            i+=1
            tokenMap[word]=i
        print(tokenMap)
        for sentence in positive:
            li = []
            for word in sentence.split(' '):
                li.append(tokenMap[word])
            pos_encoded.append(torch.tensor(li))
        for sentence in negative:
            li = []
            for word in sentence.split(' '):
                li.append(tokenMap[word])
            neg_encoded.append(torch.tensor(li))
        tensors = pos_encoded + neg_encoded
        print(tensors)
        final = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return final
        
