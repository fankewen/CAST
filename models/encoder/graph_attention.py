### graph_attention.py ###
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_attention_layout import (
    JointLengthLayout,
    build_joint_length_groups,
)
from ops.fixed_graph_attention import fixed_relation_bias


class GraphMultiHeadAttention(nn.Module):
    """
    Graph-aware multi-head attention with shortest-path-hop and edge-type
    biases, plus optional ancestor/tree masking.

    Inputs can be either static ([B,J,D] with [B,J,J] distance/edge/mask),
    or temporal ([B,T,J,D] with [B,T,J,J] distance/edge/mask). When the
    inputs carry a time axis, it is folded into the batch dimension.
    """

    def __init__(
        self,
        d_model,
        nheads=4,
        dropout=0.1,
        max_path_len=5,
        value_emb=False,
        use_tree_mask=False,
    ):
        super().__init__()
        assert d_model % nheads == 0

        self.d_model = d_model
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * self.att_size)
        self.linear_k = nn.Linear(d_model, nheads * self.att_size)
        self.linear_v = nn.Linear(d_model, nheads * self.att_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(nheads * self.att_size, d_model)
        self._inference_qkv = None

        self.max_path_len = max_path_len
        self.value_emb_flag = value_emb
        self.use_tree_mask = use_tree_mask
        self.last_backend = "uninitialized"
        self.topology_key_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_key_emb = nn.Embedding(6, d_model)
        self.topology_query_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_query_emb = nn.Embedding(6, d_model)
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, d_model)
            self.edge_value_emb = nn.Embedding(6, d_model)

    @torch.no_grad()
    def prepare_inference_fusion(self):
        if self._inference_qkv is not None:
            return
        fused = nn.Linear(self.d_model, 3 * self.d_model, bias=True).to(
            device=self.linear_q.weight.device, dtype=self.linear_q.weight.dtype
        )
        fused.weight.copy_(torch.cat((self.linear_q.weight, self.linear_k.weight, self.linear_v.weight), dim=0))
        fused.bias.copy_(torch.cat((self.linear_q.bias, self.linear_k.bias, self.linear_v.bias), dim=0))
        fused.requires_grad_(False)
        self._inference_qkv = fused

    def _project_qkv(self, q, k, v):
        self_attention = q is k and q is v
        if self.training and self_attention:
            weight = torch.cat(
                (self.linear_q.weight, self.linear_k.weight, self.linear_v.weight),
                dim=0,
            )
            bias = torch.cat(
                (self.linear_q.bias, self.linear_k.bias, self.linear_v.bias),
                dim=0,
            )
            return F.linear(q, weight, bias).chunk(3, dim=-1)
        if self._inference_qkv is not None and self_attention:
            return self._inference_qkv(q).chunk(3, dim=-1)
        return self.linear_q(q), self.linear_k(k), self.linear_v(v)

    @torch.no_grad()
    def build_streaming_relation_cache(self, distance, edge_attr):
        if distance.ndim != 3 or edge_attr.shape != distance.shape:
            raise ValueError("streaming topology must be [B,J,J]")
        q_rel = self.topology_query_emb(distance.long()) + self.edge_query_emb(edge_attr.long())
        k_rel = self.topology_key_emb(distance.long()) + self.edge_key_emb(edge_attr.long())
        shape = q_rel.shape[:-1] + (self.nheads, self.att_size)
        return {
            "query": q_rel.view(shape).permute(0, 3, 1, 2, 4).contiguous(),
            "key": k_rel.view(shape).permute(0, 3, 1, 2, 4).contiguous(),
        }

    def forward(
        self,
        q,
        k,
        v,
        distance,
        edge_attr,
        mask=None,
        tree_mask=None,
        length_groups: JointLengthLayout | None = None,
        relation_cache=None,
    ):
        if distance.dim() == 4:
            batch, frames, joints, _ = distance.shape
            distance = distance.reshape(batch * frames, joints, joints)
            edge_attr = edge_attr.reshape(batch * frames, joints, joints)
            if mask is not None:
                mask = mask.reshape(batch * frames, joints)
            if tree_mask is not None:
                tree_mask = tree_mask.reshape(
                    batch * frames, joints, joints
                )

        if mask is None:
            return self._forward_dense(
                q, k, v, distance, edge_attr, mask=None, tree_mask=tree_mask,
                relation_cache=relation_cache,
            )
        if length_groups is None:
            length_groups = build_joint_length_groups(mask)
        assert length_groups is not None
        joints = q.shape[1]
        if not length_groups.use_grouped:
            return self._forward_dense(
                q, k, v, distance, edge_attr,
                mask=mask,
                tree_mask=tree_mask,
                relation_cache=relation_cache,
            )
        if (
            len(length_groups.groups) == 1
            and length_groups.groups[0][0] == joints
        ):
            return self._forward_dense(
                q, k, v, distance, edge_attr, mask=None, tree_mask=tree_mask
                , relation_cache=relation_cache
            )

        # The batch-level layout already selected grouped execution using an
        # approximate quadratic-work estimate. Reuse its row masks in every
        # graph layer and run attention only on each valid joint prefix.
        output = None
        for valid_length, row_mask in length_groups.groups:
            if valid_length == 0:
                continue
            sub_tree_mask = (
                tree_mask[row_mask, :valid_length, :valid_length]
                if tree_mask is not None
                else None
            )
            sub_output = self._forward_dense(
                q[row_mask, :valid_length],
                k[row_mask, :valid_length],
                v[row_mask, :valid_length],
                distance[row_mask, :valid_length, :valid_length],
                edge_attr[row_mask, :valid_length, :valid_length],
                mask=None,
                tree_mask=sub_tree_mask,
                relation_cache=(
                    {name: value[row_mask, :, :valid_length, :valid_length]
                     for name, value in relation_cache.items()}
                    if relation_cache is not None else None
                ),
            )
            if output is None:
                output = sub_output.new_zeros(q.shape)
            output[row_mask, :valid_length] = sub_output
        if output is None:
            return torch.zeros_like(q)
        return output

    def _forward_dense(
        self,
        q,               # [B,J,D] or [B*T,J,D]
        k,               # [B,J,D] or [B*T,J,D]
        v,               # [B,J,D] or [B*T,J,D]
        distance,        # [B,J,J] or [B,T,J,J]
        edge_attr,       # [B,J,J] or [B,T,J,J]
        mask=None,       # [B,J] or [B,T,J]
        tree_mask=None,  # [B,J,J] or [B,T,J,J]
        relation_cache=None,
    ):
        if distance.dim() == 4:
            B, T, J, _ = distance.shape
            distance = distance.reshape(B * T, J, J)
            edge_attr = edge_attr.reshape(B * T, J, J)
            if mask is not None:
                mask = mask.reshape(B * T, J)
            if tree_mask is not None:
                tree_mask = tree_mask.reshape(B * T, J, J)

        orig_q_size = q.size()
        batch_size = q.size(0)
        d_k = self.att_size
        d_v = self.att_size

        q, k, v = self._project_qkv(q, k, v)
        q = q.view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.nheads, d_v).transpose(1, 2)

        seq_len = v.shape[2]
        num_hop_types = self.max_path_len + 1
        num_edge_types = 6

        query_hop_emb = self.topology_query_emb.weight.view(
            1, num_hop_types, self.nheads, d_k
        ).transpose(1, 2)
        query_edge_emb = self.edge_query_emb.weight.view(
            1, num_edge_types, self.nheads, d_k
        ).transpose(1, 2)
        key_hop_emb = self.topology_key_emb.weight.view(
            1, num_hop_types, self.nheads, d_k
        ).transpose(1, 2)
        key_edge_emb = self.edge_key_emb.weight.view(
            1, num_edge_types, self.nheads, d_k
        ).transpose(1, 2)

        if relation_cache is not None:
            # Fixed topology allows the four relation projections to be
            # evaluated by one fused kernel.
            relation_bias = fixed_relation_bias(q, k, relation_cache, self.scale)
        else:
            distance_index = distance.unsqueeze(1).expand(-1, self.nheads, -1, -1)
            edge_index = edge_attr.unsqueeze(1).expand(-1, self.nheads, -1, -1)

            def gathered_relation_bias(token, embedding, relation_index):
                relation_logits = torch.matmul(token, embedding.transpose(2, 3))
                return torch.gather(relation_logits, 3, relation_index)

            relation_bias = gathered_relation_bias(q, query_hop_emb, distance_index)
            relation_bias.add_(gathered_relation_bias(q, query_edge_emb, edge_index))
            relation_bias.add_(gathered_relation_bias(k, key_hop_emb, distance_index))
            relation_bias.add_(gathered_relation_bias(k, key_edge_emb, edge_index))

        if not self.value_emb_flag:
            attn_mask = relation_bias.mul(self.scale)
            if tree_mask is not None and self.use_tree_mask:
                attn_mask = attn_mask.masked_fill(
                    ~tree_mask[:, None, :, :],
                    float("-inf"),
                )
            if mask is not None:
                attn_mask = attn_mask.masked_fill(
                    ~mask[:, None, None, :],
                    float("-inf"),
                )
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
            if mask is not None:
                x = x * mask[:, None, :, None].to(x.dtype)
            self.last_backend = "sdpa"
        else:
            attn_score = (
                torch.matmul(q, k.transpose(2, 3))
                + relation_bias
            )
            attn_score = attn_score * self.scale

            if tree_mask is not None and self.use_tree_mask:
                attn_score = attn_score.masked_fill(
                    ~tree_mask[:, None, :, :], float("-inf")
                )

            if mask is not None:
                mask_k = mask[:, None, None, :]
                attn_score = attn_score.masked_fill(~mask_k, float("-inf"))

            invalid_rows = torch.isinf(attn_score).all(dim=-1, keepdim=True)
            attn_score = torch.where(
                invalid_rows,
                torch.zeros_like(attn_score),
                attn_score,
            )

            attn = torch.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)

            if mask is not None:
                mask_q = mask[:, None, :, None].to(attn.dtype)
                attn = attn * mask_q

            if tree_mask is not None and self.use_tree_mask:
                attn = attn * tree_mask[:, None, :, :].to(attn.dtype)

            attn = self.dropout(attn)
            self.last_backend = "dense"

            value_hop_emb = self.topology_value_emb.weight.view(
                1, num_hop_types, self.nheads, d_k
            ).transpose(1, 2)
            value_edge_emb = self.edge_value_emb.weight.view(
                1, num_edge_types, self.nheads, d_k
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, seq_len, num_hop_types),
                device=q.device,
                dtype=attn.dtype,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att,
                3,
                distance_index,
                attn,
            )

            value_edge_att = torch.zeros(
                (batch_size, self.nheads, seq_len, num_edge_types),
                device=q.device,
                dtype=attn.dtype,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att,
                3,
                edge_index,
                attn,
            )

            x = torch.matmul(attn, v)
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(
                value_edge_att, value_edge_emb
            )

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nheads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x
