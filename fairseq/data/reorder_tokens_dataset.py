# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary
from math import ceil

from . import BaseWrapperDataset, LRUCacheDataset

class ReorderCLDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, seed: int, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, seed=seed, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, seed=seed * 4 + 1, *args, **kwargs, return_masked_tokens=False)),
        )
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        switch_token_nums: int = 3,
        switch_token_max_prop: float = 0.05,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.switch_token_nums = switch_token_nums
        self.switch_token_max_prop = switch_token_max_prop

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        mask_candidate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            real_mask_prob = mask_candidate[np.random.choice(10, 1)[0]]
            item = self.dataset[index]
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            new_item = np.copy(item)
            tmp_switch_token_nums = np.random.choice(self.switch_token_nums + 1, 1)[0]
            if tmp_switch_token_nums == 0:
                return torch.from_numpy(new_item)

            def get_switch_index(used, prop, sz, num):
                mean_len = np.round(prop * sz)
                while True:
                    st = int(np.random.choice(sz - ceil(mean_len), 1)) + 1
                    while True:
                        randomNums = np.random.normal(loc = mean_len, scale=3, size=1)
                        tmp_len = int(np.round(randomNums))
                        
                        if tmp_len < 0 or tmp_len > sz - st:
                            continue
                        else:
                            break
                    en = st + tmp_len - 1
                    check = True
                    for i in used:
                        if i[0] > en or i[1] < st:
                            continue
                        else:
                            check = False
                            break
                    if check:
                        used.append((st, en, num))
                        return st, en
                    else:
                        continue

            rep_item = []
            # decide elements to replace
            used = np.full(sz, True)
            used = []
            for _rep in range(self.switch_token_nums):
                st1, en1 = get_switch_index(used, self.switch_token_max_prop, sz, _rep)
                st2, en2 = get_switch_index(used, self.switch_token_max_prop, sz, _rep)
            used = sorted(used, key=lambda x: x[0])
            pre = 0
            for i in used:
                for j in range(pre, i[0]):
                    rep_item.append(new_item[j])
                for k in used:
                    if k[2] == i[2] and k[0] != i[0]:
                        break
                for j in range(k[0], k[1] + 1):
                    rep_item.append(new_item[j])
                pre = i[1] + 1
            for j in range(pre, len(new_item)):
                rep_item.append(new_item[j])
            rep_item = np.array(rep_item) 

            return torch.from_numpy(rep_item)
