
# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py

import math
import copy
import numpy as np
from typing import List, Optional, Tuple, Union
from ..nn_utils.blockdiag import BlockdiagLinear
from ..nn_utils.hyena import HyenaFilter

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial

from einops import rearrange
class MonarchMixerSequenceMixing(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        dropout=0.0,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=3,
        hyena_filter_dropout=0.0,
        hyena_filter_order=16,
        residual_long_conv=False,
        hyena_training_additions=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv
        self.NUM_PROJECTIONS = 3

        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)
        print(f"-- Hyena filter order: {hyena_filter_order}")
        print(f"-- Hyena filter dropout: {hyena_filter_dropout}")
        print(f"-- Hyena filter wd: {hyena_wd}")
        print(f"-- Hyena filter emb dim: {hyena_emb_dim}")
        print(f"-- Hyena filter lr: {hyena_kernel_lr}")
        print(f"-- Hyena filter lr pos emb: {hyena_lr_pos_emb}")

        self.filter_fn = HyenaFilter(
            self.d_model,
            order=hyena_filter_order,
            seq_len=self.l_max,
            dropout=hyena_filter_dropout,
            bidirectional=self.bidirectional,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,  # frequency of periodic activations
            w_mod=hyena_w_mod,
            wd=hyena_wd,  # weight decay of kernel parameters
            emb_dim=hyena_emb_dim,
        )
        
        if self.residual_long_conv:
            self.filter_fn2 = HyenaFilter(
                self.d_model,
                order=hyena_filter_order,
                seq_len=self.l_max,
                dropout=hyena_filter_dropout,
                bidirectional=self.bidirectional,
                lr=hyena_kernel_lr,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,  # frequency of periodic activations
                w_mod=hyena_w_mod,
                wd=hyena_wd,  # weight decay of kernel parameters
                emb_dim=hyena_emb_dim,
            )
        
        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.hyena_training_additions = hyena_training_additions
        if self.hyena_training_additions:
            self.act = nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)
        
        # setup short conv
        total_width = self.d_model * self.NUM_PROJECTIONS
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=3,
            groups=total_width,
            padding=2,
        )


    def forward(self, u, **kwargs):
        # u is B L H
        if self.hyena_training_additions:
            u = self.layernorm(u)
        L = u.size(-2)

        # in projection
        u_orig = u
        u = self.in_linear(u)
        u = rearrange(u, "b l d -> b d l")
        
        # short filter
        uc = self.short_filter(u)[..., :L]

        x1, x2, v = uc.split(self.d_model, dim=1)
        
        v = v * x1
        if self.hyena_training_additions:
            v = self.drop(v)

        k = self.filter_fn.filter(L, device=u.device)
        k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default

        if self.bidirectional:
            k_rev = self.filter_fn.filter_rev(L, device=u.device)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
        else:
            k_rev = None

        y = self.filter_fn(v, L, k_fwd=k, k_rev=k_rev, bias= self.filter_fn.bias[None, :, None])

        if self.residual_long_conv:
            k2 = self.filter_fn2.filter(L, device=u.device)
            k2 = rearrange(k2, "c l d -> c d l")[0]

            if self.bidirectional:
                k2_rev = self.filter_fn2.filter_rev(L, device=u.device)
                k2_rev = rearrange(k2_rev, "c l d -> c d l")[0] # `c` is always 1 by default
            else:
                k2_rev = None                

            yu = self.filter_fn2(u_orig.transpose(-1, -2), L, k_fwd=k2, k_rev=k2_rev, bias= self.filter_fn2.bias[None, :, None])
        
        # post gating
        y = y * x2

        if self.residual_long_conv:
            y = y + yu

        y = y.transpose(-1, -2)
        if self.hyena_training_additions:
            y = self.drop(self.act(y))
        y = self.out_linear(y)

        return y, None


class BertGatedLinearUnitMLP(nn.Module):
    """Applies the FFN at the end of each BERT layer with a Gated Linear Unit"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.is_padded = config["monarch_mixer_sequence_mixing"]

        if self.config["use_monarch_mlp"]:
            linear_cls = partial(BlockdiagLinear, nblocks=self.config["monarch_mlp_nblocks"])
        else:
            linear_cls = nn.Linear
        self.gated_layers = linear_cls(
            config["hidden_size"],
            config["intermediate_size"] * 2,
            bias=False
        )
        self.act = nn.GELU(approximate='none')
        self.wo = linear_cls(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.layernorm = nn.LayerNorm(config["hidden_size"],
                                      eps=config["layer_norm_eps"])
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """

        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)

        if self.is_padded:
            gated = hidden_states[:, :, :self.config["intermediate_size"]]
            non_gated = hidden_states[:, :, self.config["intermediate_size"]:]
        else:
            gated = hidden_states[:, :self.config["intermediate_size"]]
            non_gated = hidden_states[:, self.config["intermediate_size"]:]

        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)

        return hidden_states

class BertMLP(nn.Module):
    """Applies the FFN at the end of each BERT layer"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.is_padded = config["monarch_mixer_sequence_mixing"]

        if self.config["use_monarch_mlp"]:
            linear_cls = partial(BlockdiagLinear, nblocks=self.config["monarch_mlp_nblocks"])
        else:
            linear_cls = nn.Linear
        self.gated_layers = linear_cls(
            config["hidden_size"],
            config["intermediate_size"],
            bias=False
        )
        self.act = nn.GELU(approximate='none')
        self.wo = linear_cls(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.layernorm = nn.LayerNorm(config["hidden_size"],
                                      eps=config["layer_norm_eps"])
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """

        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)

        
        gated = hidden_states
        hidden_states = self.act(gated)
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)

        return hidden_states

class BertM2Layer(nn.Module):
    """BERT layer, which includes Sequence Mixing (e.g. Attention or Hyena) and State Mixing (e.g. MLP)."""

    def __init__(self, config):
        super(BertM2Layer, self).__init__()
        
        print(f"Using Monarch Mixer for Sequence Mixing")
        mm_cls = MonarchMixerSequenceMixing
        self.attention = mm_cls(
            config["hidden_size"],
            l_max=config["long_conv_l_max"],
            hyena_kernel_lr=config["long_conv_kernel_learning_rate"],
            bidirectional=config["bidirectional"],

            hyena_lr_pos_emb=config["hyena_lr_pos_emb"],
            hyena_w=config["hyena_w"],
            hyena_w_mod=config["hyena_w_mod"],
            hyena_wd=config["hyena_wd"],
            hyena_emb_dim=config["hyena_emb_dim"],
            hyena_filter_dropout=config["hyena_filter_dropout"],
            hyena_filter_order=config["hyena_filter_order"],
            residual_long_conv=config["residual_long_conv"],
            hyena_training_additions=config["hyena_training_additions"],
        )
        self.mlp = BertGatedLinearUnitMLP(config)
        # self.mlp = BertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """

        attention_output = self.attention(hidden_states)
        if type(attention_output) == tuple:
            attention_output, _ = attention_output
        layer_output = self.mlp(attention_output)

        return layer_output


class M2BertEncoder(nn.Module):
    """A stack of BERT layers providing the backbone of BERT.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        layer = BertM2Layer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config["num_hidden_layers"])])

        self.monarch_mixer_sequence_mixing = config["monarch_mixer_sequence_mixing"]
        self.num_attention_heads = config["num_attention_heads"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
        position_encodings: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        
        batch, seqlen = hidden_states.shape[:2]
    
        cu_seqlens = None
        indices = None
        alibi_attn_mask = None
        all_encoder_layers = []
    
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states,
                cu_seqlens,
                seqlen,
                None,
                indices,
                attn_mask=attention_mask,
                bias=alibi_attn_mask
            )
            if position_encodings is not None:
                hidden_states = hidden_states + position_encodings
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if subset_mask is not None:
            hidden_states = hidden_states[subset_mask]
            
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers