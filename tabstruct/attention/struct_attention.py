
from tabstruct.attention.bias_utils import get_relative_relation_ids
from torch import nn
import torch

from typing import List, Optional, Tuple


from transformers import BartConfig


import math

class RelationAwareBias(nn.Module):
    def __init__(self, d_model, num_heads, num_relations, rank=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.rank = rank
        self.head_dim = d_model // num_heads

        # Relation-specific low-rank LoRA-style adapters for Q and K
        # Changer l'initialization 
        self.q_proj_A = nn.Parameter(torch.randn(num_relations, num_heads, self.rank, self.head_dim))
        self.q_proj_B = nn.Parameter(torch.randn(num_relations, num_heads, self.head_dim, self.rank))

        self.k_proj_A = nn.Parameter(torch.randn(num_relations, num_heads, self.rank, self.head_dim))
        self.k_proj_B = nn.Parameter(torch.randn(num_relations, num_heads, self.head_dim, self.rank))

        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.q_proj_A, self.q_proj_B, self.k_proj_A, self.k_proj_B]:
            nn.init.zeros_(param)



    def forward(self, Q, K, relation_ids):
        B, H, L, D = Q.shape
        R = self.num_relations
        X = self.rank

        Q_proj = torch.einsum("rhdx,bhld->brhlx", self.q_proj_B, Q)  # (B, R, H, L, X)
        Q_adapted = torch.einsum("rhxd,brhlx->brhld", self.q_proj_A, Q_proj)  # (B, R, H, L, D)

        # Project K
        K_proj = torch.einsum("rhdx,bhld->brhlx", self.k_proj_B, K)
        K_adapted = torch.einsum("rhxd,brhlx->brhld", self.k_proj_A, K_proj)

        scores = torch.einsum("brhld,brhmd->brhlm", Q_adapted, K_adapted)  # (B, R, H, L, L)
        scores = scores.permute(0, 2, 1, 3, 4)  # (B, H, R, L, L)

        # Gather the correct relation scores
        relation_ids_exp = relation_ids.long().unsqueeze(1).expand(-1, H, -1, -1)
        bias = torch.gather(scores, dim=2, index=relation_ids_exp.unsqueeze(2)).squeeze(2)  # (B, H, L, L)
                
        return bias


"""class RelationAwareStructureMatrix(nn.Module):
    def __init__(self, num_heads, num_relations, seq_len, head_dim, rank=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.rank = rank

        # Low-rank factors for A = I + A1 @ A2
        self.A1 = nn.Parameter(torch.zeros(num_relations, num_heads, head_dim, rank))
        self.A2 = nn.Parameter(torch.zeros(num_relations, num_heads, rank, head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.A1)
        nn.init.zeros_(self.A2)

    def forward(self, Q, K, relation_ids):

        B, H, L, D = Q.size()
        R = self.num_relations

        # Precompute A_r = I + A1 @ A2 for all relations and heads
        A_rel = torch.matmul(self.A1, self.A2)  # (R, H, D, D)
        identity = torch.eye(D, device=Q.device).unsqueeze(0).unsqueeze(0)  # (1, 1, D, D)
        A_rel = A_rel + identity  # (R, H, D, D)

        # Now we gather A matrices per (i,j) from relation_ids
        # Output shape: (B, H, L, L, D, D)
        rel_ids_exp = relation_ids.unsqueeze(1).expand(-1, H, -1, -1)  # (B, H, L, L)
        A = A_rel[rel_ids_exp]  # gather (B, H, L, L, D, D)

        # Apply QAKᵀ
        Q_exp = Q.unsqueeze(3)  # (B, H, L, 1, D)
        K_exp = K.unsqueeze(2)  # (B, H, 1, L, D)

        # Compute Q @ A: (B, H, L, L, D)
        QA = torch.matmul(Q_exp, A).squeeze(3)  # (B, H, L, L, D)
        attn_scores = torch.einsum('bhlmd,bhlnd->bhlmn', QA, K_exp)  # (B, H, L, L)

        return attn_scores"""
    

class RelationAwareStructureMatrix(nn.Module):
    def __init__(self, num_heads, num_relations, head_dim, rank=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = head_dim
        self.rank = rank

        self.A1 = nn.Parameter(torch.zeros(num_relations, num_heads, head_dim, rank))
        self.A2 = nn.Parameter(torch.zeros(num_relations, num_heads, rank, head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.A1)
        nn.init.zeros_(self.A2)

    def forward(self, Q, K, relation_ids):
        """
        Q, K: (B, H, L, D)
        relation_ids: (B, L, L)
        """
        B, H, L, D = Q.size()
        attn_scores = torch.zeros(B, H, L, L, device=Q.device, dtype=Q.dtype)

        identity = torch.eye(D, device=Q.device).unsqueeze(0).unsqueeze(0)  # (1, 1, D, D)
        A_matrices = self.A1 @ self.A2 + identity  # (R, H, D, D)

        for r in range(self.num_relations):
            A_r = A_matrices[r]  # (H, D, D)
            QA = torch.einsum('bhld,hde->bhle', Q, A_r)  # (B, H, L, D)
            scores_r = torch.einsum('bhld,bhmd->bhlm', QA, K)  # (B, H, L, L)

            # Create a mask where relation_ids == r
            rel_mask = (relation_ids == r).unsqueeze(1)  # (B, 1, L, L)
            attn_scores = attn_scores + scores_r * rel_mask

        return attn_scores  # (B, H, L, L)


class StructAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.encoding_structure_bias = config.encoding_structure_bias

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        
        if self.encoding_structure_bias == "B1":
            self.attention_bias_embeddings = nn.Embedding(13 + 1, 1)

        if self.encoding_structure_bias == "B2":
            self.relation_bias_module = RelationAwareBias(
                d_model=self.embed_dim,
                num_heads=self.num_heads,
                num_relations=13,  # match output of get_relative_relation_ids
                rank=4,           )

        if self.encoding_structure_bias == "B3":
            self.relation_bias_module = RelationAwareStructureMatrix(
                num_heads=self.num_heads,
                num_relations=13,
                head_dim=self.head_dim,
                rank=4,
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        token_type: torch.LongTensor = None,

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

      

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)


        src_len = key_states.size(1)
        
        if self.encoding_structure_bias != "B3":
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

        if attention_mask is not None:
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            
            if self.encoding_structure_bias != "B3":
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask

            if self.encoding_structure_bias == "B1":
                    
                attention_bias_ids = get_relative_relation_ids(token_type, attention_mask)
                attention_bias = self.attention_bias_embeddings(attention_bias_ids).squeeze(-1).unsqueeze(1)
                attention_bias = attention_bias.expand(-1, self.num_heads, -1, -1)
                attn_weights = attn_weights + attention_bias

            if self.encoding_structure_bias == "B2":
                # Compute relation-aware attention bias
                attention_bias_ids = get_relative_relation_ids(token_type, attention_mask)  # (B, L, L)
                # Reshape Q, K for bias module (B, num_heads, L, head_dim)
                Q = query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
                K = key_states.view(bsz, self.num_heads, src_len, self.head_dim)
                structure_bias = self.relation_bias_module(Q, K, attention_bias_ids)  # (B, num_heads, L, L)
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + structure_bias

            if self.encoding_structure_bias == "B3":
                attention_bias_ids = get_relative_relation_ids(token_type, attention_mask)  # (B, L, L)
                Q = query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
                K = key_states.view(bsz, self.num_heads, src_len, self.head_dim)
                attn_weights = self.relation_bias_module(Q, K, attention_bias_ids)  # (B, H, L, L)

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
