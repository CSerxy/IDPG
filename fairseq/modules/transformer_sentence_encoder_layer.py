# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)
from fairseq.modules.quant_noise import quant_noise


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # norm for Pfeiffer adapter
        self.middle_layer_norm = LayerNorm(self.embedding_dim, export=export)
        # gelu activation function for compacter
        self.compacter_activation_fn = utils.get_activation_fn('gelu')

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        suffix_x: Optional[torch.Tensor] = None,
        adapter_arch: str = '',
        adapter_MLP: Optional[nn.Sequential] = None,
        adapter_MLP2: Optional[nn.Sequential] = None,
        compacter_n: int = 0,
        compacter_down_proj_s: Optional[nn.ParameterList] = None,
        compacter_down_proj_t: Optional[nn.ParameterList] = None,
        compacter_down_proj_b: Optional[nn.ParameterList] = None,
        compacter_up_proj_s: Optional[nn.ParameterList] = None,
        compacter_up_proj_t: Optional[nn.ParameterList] = None,
        compacter_up_proj_b: Optional[nn.Embedding] = None,
        compacter_shared_A: Optional[nn.ParameterList] = None,
        compacter_down_proj_s2: Optional[nn.ParameterList] = None,
        compacter_down_proj_t2: Optional[nn.ParameterList] = None,
        compacter_down_proj_b2: Optional[nn.ParameterList] = None,
        compacter_up_proj_s2: Optional[nn.ParameterList] = None,
        compacter_up_proj_t2: Optional[nn.ParameterList] = None,
        compacter_up_proj_b2: Optional[nn.Embedding] = None,
        compacter_shared_A2: Optional[nn.ParameterList] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if suffix_x != None:
            key_x = torch.cat((x[:1][:], suffix_x[:suffix_x.size()[0] // 2][:], x[1:][:]), 0)
            value_x = torch.cat((x[:1][:], suffix_x[suffix_x.size()[0] // 2:][:], x[1:][:]), 0)
            if self_attn_padding_mask is not None:
                a = torch.zeros(x.size()[1], suffix_x.size()[0] // 2).type_as(self_attn_padding_mask)
                #a = torch.zeros(x.size()[1], suffix_x.size()[0] // 2, dtype=torch.bool).to('cuda:0')
                x, attn = self.self_attn(
                    query=x,
                    key=key_x,
                    value=value_x,
                    key_padding_mask=torch.cat((self_attn_padding_mask[:, :1], a, self_attn_padding_mask[:, 1:]), 1),
                    need_weights=False,
                    attn_mask=self_attn_mask,
                )
            else:
                x, attn = self.self_attn(
                    query=x,
                    key=key_x,
                    value=value_x,
                    key_padding_mask=self_attn_padding_mask,
                    need_weights=False,
                    attn_mask=self_attn_mask,
                )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        if adapter_arch == 'houlsby':
            residual_adapter = x

            x = adapter_MLP2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = residual_adapter + x

        if adapter_arch == 'compacter':
            residual_adapter = x

            #x = adapter_MLP(x)
            xt = x.transpose(0, 1).reshape(-1, x.size()[2])
            # 16 x len x 1024
            tmp_x = compacter_down_proj_b2.repeat(x.size()[1], x.size()[0], 1).type_as(x)
            # 16 x len x 16
            for i in range(compacter_n):
                w = torch.kron(compacter_shared_A2[i], torch.mm(compacter_down_proj_s2[i], compacter_down_proj_t2[i]))
                wx = torch.mm(xt, w).view(x.size()[1], -1, w.size()[1])
                tmp_x += wx
            tmp_x = self.compacter_activation_fn(tmp_x)
            tmp_x = F.dropout(tmp_x, p=self.dropout, training=self.training)
            tmp_x = tmp_x.reshape(-1, tmp_x.size()[2])
            # 16len x 16

            x = compacter_up_proj_b2.view(-1).repeat(residual_adapter.size()[1], residual_adapter.size()[0], 1).type_as(tmp_x).reshape(-1, residual_adapter.size()[2])
            # 16len x 1024
            for i in range(compacter_n):
                w = torch.kron(compacter_shared_A2[i], torch.mm(compacter_up_proj_s2[i], compacter_up_proj_t2[i]))
                # 16 x 1024
                x += torch.mm(tmp_x, w).type_as(tmp_x)

            x = x.view(residual_adapter.size()[1], -1, residual_adapter.size()[2])
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 16 x len x 1024

            x = x.transpose(0, 1)
            # len x 16 x 1024
            x = residual_adapter + x

        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if adapter_arch == 'houlsby':
            residual_adapter = x

            x = adapter_MLP(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = residual_adapter + x

        if adapter_arch == 'compacter':
            residual_adapter = x

            #x = adapter_MLP(x)
            xt = x.transpose(0, 1).reshape(-1, x.size()[2])
            # 16 x len x 1024
            tmp_x = compacter_down_proj_b.repeat(x.size()[1], x.size()[0], 1).type_as(x)
            # 16 x len x 16
            for i in range(compacter_n):
                w = torch.kron(compacter_shared_A[i], torch.mm(compacter_down_proj_s[i], compacter_down_proj_t[i]))
                wx = torch.mm(xt, w).view(x.size()[1], -1, w.size()[1])
                tmp_x += wx
            tmp_x = self.compacter_activation_fn(tmp_x)
            tmp_x = F.dropout(tmp_x, p=self.dropout, training=self.training)
            tmp_x = tmp_x.reshape(-1, tmp_x.size()[2])
            # 16len x 16

            x = compacter_up_proj_b.view(-1).repeat(residual_adapter.size()[1], residual_adapter.size()[0], 1).type_as(tmp_x).reshape(-1, residual_adapter.size()[2])
            # 16len x 1024
            for i in range(compacter_n):
                w = torch.kron(compacter_shared_A[i], torch.mm(compacter_up_proj_s[i], compacter_up_proj_t[i]))
                # 16 x 1024
                x += torch.mm(tmp_x, w).type_as(tmp_x)
            x = x.view(residual_adapter.size()[1], -1, residual_adapter.size()[2])
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 16 x len x 1024

            x = x.transpose(0, 1)
            # len x 16 x 1024
            x = residual_adapter + x

        if adapter_arch == 'pfeiffer':
            residual_adapter = x

            x = residual + x
            x = self.middle_layer_norm(x)
            
            x = adapter_MLP(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = residual_adapter + x

        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
