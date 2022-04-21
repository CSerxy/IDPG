# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import math
import torch

from fairseq.data import data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset


class DeclutrDataset(BaseWrapperDataset):
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
        return cls(dataset, seed=seed, *args, **kwargs)

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        seed: int = 1,
        mask_whole_words: torch.Tensor = None,
        l_min: int = 32,
        l_max: int = 512,
    ):
        self.dataset = dataset
        self.vocab = vocab
        self.seed = seed
        self.mask_whole_words = mask_whole_words
        self.def_min = l_min
        self.def_max = l_max
        self._st_anchor = None
        self._en_anchor = None

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # in original paper, all sz >= 2048, so they do not worry 
            # about the case sz < self.l_max or sz < self.l_min
            self.l_min = min(sz, self.def_min)
            self.l_max = min(sz, self.def_max)

            l_anchor = math.floor(np.random.beta(4, 2) * (self.l_max - self.l_min) + self.l_min)
            st = np.random.randint(sz - l_anchor + 1, size=1)[0]
            en = st + l_anchor
            self._st_anchor = st
            self._en_anchor = en

            if self.mask_whole_words is not None:
                mask = np.full(sz, False)
                mask[np.arange(st, en)] = True
                mask = np.repeat(mask, word_lens)
                new_item = []
                for i in range(len(mask)):
                    if mask[i]:
                        new_item.append(item[i])
                new_item = np.array(new_item)
            else:
                new_item = np.copy(item[st:en])
            
            return torch.from_numpy(new_item)

    @property 
    def st_anchor(self):
        return self._st_anchor

    @property
    def en_anchor(self):
        return self._en_anchor

class AugmentationDeclutrDataset(DeclutrDataset):
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
    def apply_mask(cls, dataset: torch.utils.data.Dataset, anchor: DeclutrDataset, seed: int, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, anchor, seed=seed * 4 - 1, *args, **kwargs)), 
            LRUCacheDataset(cls(dataset, anchor, seed=seed * 4 + 1, *args, **kwargs)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        anchor: DeclutrDataset,
        vocab: Dictionary,
        seed: int = 1,
        mask_whole_words: torch.Tensor = None,
        l_min: int = 32,
        l_max: int = 512,
    ):
        super().__init__(dataset, vocab, seed, mask_whole_words, l_min, l_max)
        self.anchor = anchor

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # in original paper, all sz >= 2048, so they do not worry 
            # about the case sz < self.l_max or sz < self.l_min

            l_pos = math.floor(np.random.beta(2, 4) * (self.anchor.l_max - self.anchor.l_min) + self.anchor.l_min)
            while l_pos > self.anchor.en_anchor - self.anchor.st_anchor:
                l_pos = math.floor(np.random.beta(2, 4) * (self.anchor.l_max - self.anchor.l_min) + self.anchor.l_min)
            #assert self.st_anchor - l_pos >= 0
            st = np.random.randint(max(self.anchor.st_anchor - l_pos, 0), min(self.anchor.en_anchor + 1, sz - l_pos + 1), size=1)[0]
            en = st + l_pos
                
            #assert en > sz
            if en > sz:
                st1 = np.random.randint(sz - l_pos + 1, size=1)[0]
                en1 = st1 + l_pos

                print(self.anchor.st_anchor, self.anchor.en_anchor, '\t', st, en, st1, en1, sz)
                print('***********************')
                exit(0)


            if self.mask_whole_words is not None:
                mask = np.full(sz, False)
                mask[np.arange(st, en)] = True
                mask = np.repeat(mask, word_lens)
                new_item = []
                for i in range(len(mask)):
                    if mask[i]:
                        new_item.append(item[i])
                new_item = np.array(new_item)
            else:
                new_item = np.copy(item[st:en])
            
            return torch.from_numpy(new_item)
