# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
import json

from fairseq.data import data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset


class SubsTokensDataset(BaseWrapperDataset):
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
        subs_prob: probability of replacing a token with *mask_idx*.
        leave_unreplaced_prob: probability that a masked token is unmasked.
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
        subs: dict,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        subs_prob: float = 0.2,
        leave_unreplaced_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        switch_token_nums: int = 3,
        switch_token_max_prop: float = 0.05,
    ):
        assert 0.0 <= subs_prob <= 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unreplaced_prob <= 1.0
        assert random_token_prob + leave_unreplaced_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.subs_prob = subs_prob
        self.leave_unreplaced_prob = leave_unreplaced_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.switch_token_nums = switch_token_nums
        self.switch_token_max_prop = switch_token_max_prop
        self.subs = subs

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
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            # decide elements to substitute
            new_item = []
            for i in item:
                new_item.append(str(int(i)))
            switch = []
            tmp = 0
            for i in range(sz):
                if i >= tmp: 
                    for len_token in range(min(4, sz - i), 0, -1):
                        potential_token = " ".join(new_item[i:i+len_token])
                        if potential_token in self.subs:
                            switch.append((i, i + len_token - 1, self.subs[potential_token]))
                            tmp = i + len_token
                            break

            num_mask = int(
                # add a random number for probabilistic rounding
                self.subs_prob * sz + np.random.rand()
            )
            if len(switch) < num_mask:
                valid_idxs = [i for i in range(len(switch))]
            else:
                valid_idxs = np.random.choice(len(switch), num_mask, replace=False)
                valid_idxs = sorted(valid_idxs)

            rep_item = torch.LongTensor([])
            pre = 0
            count = 0
            diff = 512 - len(item)
            for valid_idx in valid_idxs:
                tmp = switch[valid_idx]
                rand_pick = np.random.choice(len(tmp[2]), 1)[0]
                if len(tmp[2][rand_pick]) - (1 + tmp[1] - tmp[0]) + count <= diff:
                    rep_item = torch.cat((rep_item, item[pre: tmp[0]], torch.LongTensor(tmp[2][rand_pick])), 0)
                    count += len(tmp[2][rand_pick]) - (1 + tmp[1] - tmp[0])
                else:
                    rep_item = torch.cat((rep_item, item[pre:tmp[1] + 1]), 0)
                pre = tmp[1] + 1
            rep_item = torch.cat((rep_item, item[pre:]), 0)

            assert len(rep_item) - len(item) == count
            #print(len(rep_item), len(item), count, len(switch), num_mask)
            return rep_item
