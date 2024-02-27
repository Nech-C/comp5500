#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden):
        """Self-attention module which computes softmax(xQ @ xK^T) @ xV

        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
        """
        super().__init__()
        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.scale = hidden ** 0.5

    def forward(self, x):
        """Softmax(xQ @ xK^T) @ xV

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        # YOUR CODE STARTS HERE (can be implemented in ~4 lines)
        q = self.q(x)  # [batch_size, seq_len, hidden]
        scores = torch.bmm(q, self.k(x).transpose(1, 2)) / self.scale
        # [batch_size, seq_len, hidden] -> [batch_size, seq_len, seq_len]
        probs = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        return torch.bmm(probs, self.v(x))
        # [batch_size, seq_len, seq_len]
        # @ [batch_size, seq_len, hidden] -> [batch_size, seq_len, hidden]
        # YOUR CODE ENDS HERE


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0):
        """
        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
            num_heads: int, number of attention heads, should be a divisor of hidden
            causal: use causal masking
            (do not allow target to look to the future or current token of source)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, x, return_attention=False):
        """Computes [Softmax(x Q_1 @ x K_1^T) @ x V_1 
        : ... : Softmax(x Q_heads @ x K_heads^T) @ x V_heads] @ U

        or in more details:
        [SelfAttention_1(x) : ... : SelfAttention_h(x)] @ U

        where SelfAttention(x) = Softmax(x Q @ x K^T) @ x V
        and [:] is a concatenation operation.

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        bs, seq, _ = x.shape

        # YOUR CODE STARTS HERE
        k = self.k(x).view(bs, seq, self.num_heads, self.head_size).transpose(1, 2)
        # shape [batch_size, num_heads, seq_len, head_size]
        # 'k' [batch_size, seq_len, input_size]
        # -> view [batch_size, seq_len, num_heads, head_size]
        # -> transpose [batch_size, num_heads, seq_len, head_size]
        q = self.q(x).view(bs, seq, self.num_heads, self.head_size).transpose(1, 2)
        # shape [batch_size, num_heads, seq_len, head_size]
        # 'q' [batch_size, seq_len, input_size]
        # -> view [batch_size, seq_len, num_heads, head_size]
        # -> transpose [batch_size, num_heads, seq_len, head_size]
        v = self.v(x).view(bs, seq, self.num_heads, self.head_size).transpose(1, 2)
        # shape [batch_size, num_heads, seq_len, head_size]
        # 'v' [batch_size, seq_len, input_size]
        # -> view [batch_size, seq_len, num_heads, head_size]
        # -> transpose [batch_size, num_heads, seq_len, head_size]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # shape [batch_size, num_heads, seq_len, seq_len]
        # 'q' [batch_size, num_heads, seq_len, head_size]
        # @ 'k.T' [batch_size, num_heads, head_size, seq_len]
        # -> [batch_size, num_heads, seq_len, seq_len]

        if self.causal:
            mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
            # [seq_len, seq_len]
            scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))
            # [bs, num_heads, seq_len, seq_len]

        probs = F.softmax(scores, dim=-1) # [bs, num_heads, seq_len, seq_len]
        probs = self.dropout(probs) # [bs, num_heads, seq_len, seq_len]

        att = torch.matmul(probs, v)
        # [bs, num_heads, seq_len, seq_len]
        # @ [bs, num_heads, seq_len, head_size] -> [bs, num_heads, seq_len, head_size]

        probs = probs.reshape(bs * self.num_heads, seq, seq)
        # 'probs' [bs, num_heads, seq_len, seq_len]
        # -> reshape [bs * num_heads, seq_len, seq_len]

        att = att.transpose(1, 2).contiguous().view(bs, seq, -1) # [bs, seq_len, hidden]

        att = self.mix(att)
        # YOUR CODE ENDS HERE

        if return_attention:
            return att, probs

        return att
