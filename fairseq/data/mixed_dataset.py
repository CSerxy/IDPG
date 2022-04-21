# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
import json

from fairseq.data import data_utils, Dictionary
from math import ceil

from . import BaseWrapperDataset, LRUCacheDataset


def eliminating(new_item, mask_idx):
    start = 0
    ret_item = []
    total_len = len(new_item)
    for i in range(total_len):
        if i >= start:
            if new_item[i] != mask_idx:
                ret_item.append(new_item[i])
            else:
                start = i
                ret_item.append(mask_idx)
                while start < total_len and new_item[start] == mask_idx:
                    start += 1
    ret_item = np.array(ret_item)

    return torch.from_numpy(ret_item)

class MixedDataset(BaseWrapperDataset):
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
        subs: dict,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        cl_mask_prob: float = 0.0,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        switch_token_nums: int = 0,
        switch_token_max_prop: float = 0.05,
        del_span_nums: int = 0,
        del_span_max_prop: float = 0.05,
        eliminate: bool = False,
        subs_prob: float = 0.0,
    ):
        assert 0.0 <= subs_prob <= 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.cl_mask_prob = cl_mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.switch_token_nums = switch_token_nums
        self.switch_token_max_prop = switch_token_max_prop
        self.del_span_nums = del_span_nums
        self.del_span_max_prop = del_span_max_prop
        self.eliminate = eliminate
        self.subs = subs
        self.subs_prob = subs_prob

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

    
    """ decide elements to substitute """
    def get_subs(self, item, sz, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
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

    def get_del_tokens(self, item, sz, index, word_lens):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.cl_mask_prob * sz + np.random.rand()
            )
            mask[np.random.choice(sz, num_mask, replace=False)] = True

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            if self.eliminate:
                return eliminating(new_item, self.mask_idx)
            else:
                return torch.from_numpy(new_item)

    def get_switch_index(self, used, prop, sz, num):
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

    def get_reorder(self, item, sz, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            new_item = np.copy(item)

            rep_item = []
            # decide elements to replace
            used = []
            for _rep in range(self.switch_token_nums):
                st1, en1 = self.get_switch_index(used, self.switch_token_max_prop, sz, _rep)
                st2, en2 = self.get_switch_index(used, self.switch_token_max_prop, sz, _rep)
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

    def get_del_span(self, item, sz, index, word_lens):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            # decide elements to replace
            used = []
            for _rep in range(self.del_span_nums):
                st1, en1 = self.get_switch_index(used, self.del_span_max_prop, sz, _rep)
                st2, en2 = self.get_switch_index(used, self.del_span_max_prop, sz, _rep)
            used = sorted(used, key=lambda x: x[0])

            mask = np.full(sz, False)
            for i in used:
                for j in range(i[0], i[1] + 1):
                    mask[j] = True

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            if self.eliminate:
                return eliminating(new_item, self.mask_idx)
            else:
                return torch.from_numpy(new_item)

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        item = self.dataset[index]
        sz = len(item)

        assert self.mask_idx not in item, \
            'Dataset contains mask_idx (={}), this is not expected!'.format(
                self.mask_idx,
            )

        word_lens = []

        if self.mask_whole_words is not None:
            word_begins_mask = self.mask_whole_words.gather(0, item)
            word_begins_idx = word_begins_mask.nonzero().view(-1)
            sz = len(word_begins_idx)
            words = np.split(word_begins_mask, word_begins_idx)[1:]
            assert len(words) == sz
            word_lens = list(map(len, words))

        if self.subs_prob != 0:
            item = self.get_subs(item, sz, index)
            sz = len(item)

        if self.cl_mask_prob != 0:
            item = self.get_del_tokens(item, sz, index, word_lens)
            sz = len(item)

        if self.del_span_nums != 0:
            item = self.get_del_span(item, sz, index, word_lens)
            sz = len(item)

        if self.switch_token_nums != 0:
            item = self.get_reorder(item, sz, index)

        return item

