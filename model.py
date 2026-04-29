# -*- coding: utf-8 -*-
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATv2Conv as GATConv, GraphConv
import pandas as pd
import scipy.spatial.distance as dist
import numpy as np

try:
    from torch.utils.checkpoint import checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False


class WeightedGraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.elu,
                 use_checkpoint=True, chunk_size=1000):
        super(WeightedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_checkpoint = use_checkpoint
        self.chunk_size = chunk_size

        self.base_conv = GraphConv(
            in_features, out_features,
            norm='both', bias=True, activation=None,
            allow_zero_in_degree=True
        )

        hidden_dim = max(out_features // 4, 32)
        self.edge_attention = nn.Sequential(
            nn.Linear(out_features * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.feature_enhance = nn.Sequential(
            nn.Linear(out_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
            nn.Dropout(0.05)
        )

        self.gate = nn.Sequential(
            nn.Linear(out_features * 2, out_features // 2),
            nn.Tanh(),
            nn.Linear(out_features // 2, 1),
            nn.Sigmoid()
        )

    def _compute_edge_weights(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            num_edges = graph.num_edges()

            if num_edges <= self.chunk_size:
                def attn_fn(edges):
                    return {'attn': self.edge_attention(
                        torch.cat([edges.src['h'], edges.dst['h']], dim=1)
                    )}
                graph.apply_edges(attn_fn)
                return graph.edata['attn']
            else:
                edges_src, edges_dst = graph.edges()
                edge_weights = []
                for i in range(0, num_edges, self.chunk_size):
                    end = min(i + self.chunk_size, num_edges)
                    ef = torch.cat([h[edges_src[i:end]], h[edges_dst[i:end]]], dim=1)
                    edge_weights.append(self.edge_attention(ef))
                return torch.cat(edge_weights, dim=0)

    def _attention_forward(self, graph, h_base):
        with graph.local_scope():
            edge_weights = self._compute_edge_weights(graph, h_base)
            graph.ndata['h'] = h_base
            graph.edata['w'] = F.softmax(edge_weights, dim=0)
            graph.update_all(
                lambda edges: {'m': edges.src['h'] * edges.data['w']},
                lambda nodes: {'h_attn': torch.sum(nodes.mailbox['m'], dim=1)}
            )
            return graph.ndata['h_attn']

    def forward(self, graph, features):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        h_base = self.base_conv(graph, features)

        if self.use_checkpoint and self.training:
            h_attn = torch.utils.checkpoint.checkpoint(
                self._attention_forward, graph, h_base
            )
        else:
            h_attn = self._attention_forward(graph, h_base)

        h_enhanced = self.feature_enhance(h_base)
        gate_w = self.gate(torch.cat([h_base, h_attn], dim=-1))
        h = h_base + gate_w * (h_attn + h_enhanced) * 0.15

        if self.activation is not None:
            h = self.activation(h)
        return h


class LightGraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.elu):
        super(LightGraphConv, self).__init__()
        self.activation = activation

        self.base_conv = GraphConv(
            in_features, out_features,
            norm='both', bias=True, activation=None,
            allow_zero_in_degree=True
        )

        hidden_dim = max(out_features // 4, 32)
        self.feature_enhance = nn.Sequential(
            nn.Linear(out_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
            nn.Dropout(0.05)
        )

        self.gate = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, features):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        h_base = self.base_conv(graph, features)
        h_enhanced = self.feature_enhance(h_base)
        gate_w = self.gate(h_base)
        h = h_base + gate_w * h_enhanced * 0.1

        if self.activation is not None:
            h = self.activation(h)
        return h


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.elu,
                 size_threshold=5000):
        super(AdaptiveGraphConv, self).__init__()
        self.size_threshold = size_threshold

        self.full_conv = WeightedGraphConv(
            in_features, out_features, activation, use_checkpoint=True
        )
        self.lite_conv = LightGraphConv(
            in_features, out_features, activation
        )

    def forward(self, graph, features):
        if (graph.num_nodes() > self.size_threshold or
                graph.num_edges() > self.size_threshold * 10):
            return self.lite_conv(graph, features)
        else:
            return self.full_conv(graph, features)


class GatedFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_topo, h_sem):
        concat  = torch.cat([h_topo, h_sem], dim=-1)   # [N, 2D]
        weights = self.gate(concat)                      # [N, 2]
        h_fused = weights[:, 0:1] * h_topo + weights[:, 1:2] * h_sem
        return h_fused


class MetaPathEncoder(nn.Module):

    def __init__(self, meta_paths, inpsize, oupsize, layer_num_heads=1,
                 dropout=0.2, types="", conv_type="adaptive",
                 alpha_g=0.2, alpha_n=0.2):
        super(MetaPathEncoder, self).__init__()

        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.oupsize = oupsize
        self.meta_paths = list(tuple(mp) for mp in meta_paths)
        self.graph_type = "heterogeneous" if len(meta_paths) > 1 else "homogeneous"

        self.gat_layers = nn.ModuleList()
        for _ in range(len(meta_paths)):
            if conv_type == "adaptive":
                conv = AdaptiveGraphConv(inpsize, oupsize, activation=F.elu)
            elif conv_type == "memory_efficient":
                conv = WeightedGraphConv(inpsize, oupsize, activation=F.elu)
            else:
                conv = LightGraphConv(inpsize, oupsize, activation=F.elu)
            self.gat_layers.append(conv)

        self.semantic_W = nn.Linear(oupsize, oupsize)
        self.semantic_b = nn.Parameter(torch.zeros(1, oupsize))
        self.semantic_q = nn.Linear(oupsize, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _mask_edges(self, graph, mask_ratio):
        if mask_ratio <= 0.0 or graph.num_edges() == 0:
            return graph

        num_edges = graph.num_edges()
        num_to_remove = int(num_edges * mask_ratio)
        mask_indices = torch.randperm(num_edges, device=graph.device)[:num_to_remove]

        try:
            src, _ = graph.edges()
            mask_indices = mask_indices.to(src.dtype)
            return dgl.remove_edges(graph, mask_indices)
        except Exception as e:
            for dtype in [torch.int64, torch.int32]:
                try:
                    return dgl.remove_edges(graph, mask_indices.to(dtype))
                except Exception:
                    continue
            print(f"[Warning] Edge masking failed, using original graph. Error: {e}")
            return graph

    def forward(self, g, h):
        semantic_embeddings = []

        masked_graph_index = -1
        if (self.training and self.graph_type == "heterogeneous"
                and len(self.meta_paths) > 1):
            if torch.rand(1).item() < self.alpha_g:
                masked_graph_index = torch.randint(0, len(self.meta_paths), (1,)).item()

        for i, paths in enumerate(self.meta_paths):
            if i == masked_graph_index:
                zero_emb = torch.zeros((h.shape[0], self.oupsize), device=h.device)
                semantic_embeddings.append(zero_emb)
                continue

            curr_graph = g[paths]

            if self.training and self.alpha_n > 0:
                curr_graph = self._mask_edges(curr_graph, self.alpha_n)

            emb = self.gat_layers[i](curr_graph, h)
            semantic_embeddings.append(emb)

        stack_emb = torch.stack(semantic_embeddings, dim=1)

        if self.graph_type == "heterogeneous":
            trans_emb = torch.tanh(self.semantic_W(stack_emb) + self.semantic_b)
            graph_summary = torch.mean(trans_emb, dim=0)
            importance = self.semantic_q(graph_summary)
            weights = F.softmax(importance, dim=0)
            fused_emb = (stack_emb * weights.unsqueeze(0)).sum(dim=1)
            return self.dropout(fused_emb)
        else:
            return stack_emb.squeeze(1)


class Classifier(nn.Module):
    def __init__(self, nfeat):
        super(Classifier, self).__init__()
        self.L1 = nn.Linear(nfeat, nfeat * 2)
        self.L2 = nn.Linear(nfeat * 2, 2)

    def forward(self, x):
        out = nn.ELU()(self.L1(x))
        return self.L2(out)


class MFMA_DTI(nn.Module):
    def __init__(self,
                 all_meta_paths,
                 in_size,
                 hidden_size,
                 out_size,
                 dropout=0.2,
                 layersnums=1,
                 att_heads=1,
                 conv_type="adaptive",
                 batch_size_limit=None,
                 feature_dims=None,
                 alpha_g=0.2,
                 alpha_n=0.2):
        super(MFMA_DTI, self).__init__()

        self.batch_size_limit = batch_size_limit

        self.meta_path_encoders = nn.ModuleList()
        for i in range(len(all_meta_paths)):
            self.meta_path_encoders.append(
                MetaPathEncoder(
                    all_meta_paths[i], in_size, hidden_size,
                    dropout=dropout, conv_type=conv_type,
                    alpha_g=alpha_g, alpha_n=alpha_n
                )
            )

        self.feature_projections = nn.ModuleList()
        if feature_dims is not None:
            self.feature_projections.append(nn.Sequential(
                nn.Linear(feature_dims[0], hidden_size),
                nn.ELU(),
                nn.Dropout(dropout)
            ))
            self.feature_projections.append(nn.Sequential(
                nn.Linear(feature_dims[1], hidden_size),
                nn.ELU(),
                nn.Dropout(dropout)
            ))
        else:
            self.feature_projections.append(nn.Identity())
            self.feature_projections.append(nn.Identity())

        self.drug_fusion = GatedFusion(hidden_size, dropout)
        self.prot_fusion = GatedFusion(hidden_size, dropout)
        node_dim = hidden_size
        dpp_dim  = node_dim * 2

        self.encoder = nn.ModuleList()
        if layersnums >= 2:
            self.encoder.append(GATConv(
                dpp_dim, hidden_size * 2,
                attn_drop=dropout, feat_drop=dropout,
                num_heads=1, residual=True, activation=F.elu,
                allow_zero_in_degree=True
            ))
            for _ in range(layersnums - 2):
                self.encoder.append(GATConv(
                    hidden_size * 2, hidden_size * 2,
                    attn_drop=dropout, feat_drop=dropout,
                    num_heads=att_heads, residual=True, activation=F.elu,
                    allow_zero_in_degree=True
                ))
            self.encoder.append(GATConv(
                hidden_size * 2, out_size * 2,
                attn_drop=dropout, feat_drop=dropout,
                num_heads=1, residual=True, activation=None,
                allow_zero_in_degree=True
            ))
        else:
            self.encoder.append(GATConv(
                dpp_dim, out_size * 2,
                attn_drop=dropout, feat_drop=dropout,
                num_heads=1, residual=True, activation=F.elu,
                allow_zero_in_degree=True
            ))

        self.topology_encoder = nn.ModuleList()
        self.semantic_encoder  = nn.ModuleList()
        for layer_idx in range(layersnums):
            in_dim  = dpp_dim       if layer_idx == 0              else hidden_size * 2
            out_dim = out_size * 2  if layer_idx == layersnums - 1 else hidden_size * 2
            act     = None          if layer_idx == layersnums - 1 else F.elu

            self.topology_encoder.append(GATConv(
                in_dim, out_dim, attn_drop=dropout, feat_drop=dropout,
                num_heads=1, residual=True, activation=act,
                allow_zero_in_degree=True
            ))
            self.semantic_encoder.append(GATConv(
                in_dim, out_dim, attn_drop=dropout, feat_drop=dropout,
                num_heads=1, residual=True, activation=act,
                allow_zero_in_degree=True
            ))

        self.projection_head = nn.Sequential(
            nn.Linear(out_size * 2, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size // 2)
        )

        self.layer_nums = layersnums
        self.predict = Classifier(out_size * 2)

    def forward(self, s_g, random_embed, pretrained_embed, data, ind,
                iftrain=True, e=1):
        if self.batch_size_limit and len(data) > self.batch_size_limit:
            return self._forward_in_batches(
                s_g, random_embed, pretrained_embed, data, ind, iftrain, e
            )
        return self._forward_single_batch(
            s_g, random_embed, pretrained_embed, data, ind, iftrain, e
        )

    def _forward_single_batch(self, s_g, random_embed, pretrained_embed, data,
                              ind, iftrain=True, e=1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        topo_drug = self.meta_path_encoders[0](s_g[0], random_embed[0])   # [N_d, D]
        topo_prot = self.meta_path_encoders[1](s_g[1], random_embed[1])   # [N_p, D]


        sem_drug = self.feature_projections[0](pretrained_embed[0])        # [N_d, D]
        sem_prot = self.feature_projections[1](pretrained_embed[1])        # [N_p, D]


        drug_final = self.drug_fusion(topo_drug, sem_drug)                 # [N_d, D]
        prot_final = self.prot_fusion(topo_prot, sem_prot)                 # [N_p, D]

        DPP_emb = torch.cat(
            (drug_final[data[:, 0]], prot_final[data[:, 1]]), dim=-1
        )

        return self._tsc_forward(DPP_emb, data, iftrain)

    def _forward_in_batches(self, s_g, random_embed, pretrained_embed, data,
                            ind, iftrain=True, e=1):
        batch_size = self.batch_size_limit
        all_outputs = []
        total_loss = 0.0

        for i in range(0, len(data), batch_size):
            batch_data = data[i: i + batch_size]
            out, loss = self._forward_single_batch(
                s_g, random_embed, pretrained_embed,
                batch_data, ind, iftrain, e
            )
            all_outputs.append(out)
            total_loss += loss * len(batch_data)

        return torch.cat(all_outputs, dim=0), total_loss / len(data)

    def _tsc_forward(self, DPP_emb, data, iftrain=True):
        topology_graph = self.build_topology_graph(data[:, :2])
        semantic_graph = self.build_semantic_graph(DPP_emb)

        topo_dgl = self._build_dgl_graph(topology_graph, DPP_emb.size(0)).to(DPP_emb.device)
        sem_dgl  = self._build_dgl_graph(semantic_graph,  DPP_emb.size(0)).to(DPP_emb.device)

        h_topo = DPP_emb
        for layer in self.topology_encoder:
            h_topo = layer(topo_dgl, h_topo).squeeze()

        h_semantic = DPP_emb
        for layer in self.semantic_encoder:
            h_semantic = layer(sem_dgl, h_semantic).squeeze()

        if iftrain:
            labels = data[:, 2]
            contrast_loss = self._compute_tsc_loss(
                h_topo=h_topo, h_semantic=h_semantic,
                topo_graph=topo_dgl, sem_graph=sem_dgl,
                DPP_emb=DPP_emb, data=data, labels=labels,
                temperature=0.1, lambda_weight=0.5
            )
        else:
            contrast_loss = torch.tensor(0.0, device=DPP_emb.device)

        h_final = (h_topo + h_semantic) / 2
        out = self.predict(h_final)

        return out, contrast_loss

    def _compute_tsc_loss(self, h_topo, h_semantic, topo_graph, sem_graph,
                          DPP_emb, data, labels, temperature=0.1, lambda_weight=0.5):
        z_topo = F.normalize(self.projection_head(h_topo), p=2, dim=1)
        z_sem  = F.normalize(self.projection_head(h_semantic), p=2, dim=1)

        Lt = self._contrastive_loss_one_direction(
            z_topo, z_sem, sem_graph, DPP_emb, data, labels, temperature
        )
        Ls = self._contrastive_loss_one_direction(
            z_sem, z_topo, topo_graph, DPP_emb, data, labels, temperature
        )

        return lambda_weight * Lt + (1 - lambda_weight) * Ls

    def _contrastive_loss_one_direction(self, z_source, z_target, target_graph,
                                        DPP_emb, data, labels, temperature):
        batch_size = z_source.size(0)
        device = z_source.device

        pos_pairs, neg_pairs = self._build_pairs_simplified(
            topo_graph=target_graph,
            sem_embeddings=DPP_emb,
            data=data,
            labels=labels,
            batch_size=batch_size,
            device=device,
            feature_sim_threshold=0.6,
            neg_feature_threshold=0.4
        )

        total_loss = 0.0
        valid_nodes = 0

        for i in range(batch_size):
            if i not in pos_pairs or len(pos_pairs[i]) <= 1:
                continue

            zi = z_source[i].unsqueeze(0)
            pos_idx = [p for p in pos_pairs[i] if p != i]
            neg_idx = neg_pairs[i]

            if not pos_idx or not neg_idx:
                continue

            pos_target = z_target[torch.tensor(pos_idx, device=device)]
            neg_target = z_target[torch.tensor(neg_idx, device=device)]

            # 正样本混合负样本（固定启用）
            k = min(20, len(neg_idx))
            cand_idx = (neg_idx[:k] if len(neg_idx) <= k else
                        [neg_idx[p] for p in torch.randperm(len(neg_idx))[:k].tolist()])
            z_cand = z_target[torch.tensor(cand_idx, device=device)]

            alpha    = torch.rand(k, 1, device=device) * 0.4
            z_mixed  = alpha * zi.expand(k, -1) + (1 - alpha) * z_cand
            z_mixed  = F.normalize(z_mixed, p=2, dim=1)

            hardest    = torch.argmax(torch.mm(zi, z_mixed.t()), dim=1).item()
            z_hard_neg = z_mixed[hardest].unsqueeze(0)

            pos_sim  = torch.mm(zi, pos_target.t()) / temperature
            neg_sim  = torch.mm(zi, neg_target.t()) / temperature
            hard_sim = torch.mm(zi, z_hard_neg.t()) / temperature
            all_sim  = torch.cat([pos_sim, neg_sim, hard_sim], dim=1)

            sim_max, _ = torch.max(all_sim, dim=1, keepdim=True)
            pos_exp = torch.exp(pos_sim - sim_max).sum()
            all_exp = torch.exp(all_sim - sim_max).sum()

            if pos_exp > 0 and all_exp > 0:
                total_loss += -torch.log(pos_exp / all_exp)
                valid_nodes += 1

        return total_loss / valid_nodes if valid_nodes > 0 else torch.tensor(0.0, device=device)

    def _build_pairs_simplified(self, topo_graph, sem_embeddings, data, labels,
                                batch_size, device,
                                feature_sim_threshold=0.6,
                                neg_feature_threshold=0.4):
        src, dst = topo_graph.edges()
        valid = (src < batch_size) & (dst < batch_size)
        src, dst = src[valid], dst[valid]

        adj = torch.zeros((batch_size, batch_size), device=device)
        adj[src, dst] = 1.0
        adj = (adj + adj.t()).clamp(0, 1)

        sem_norm = F.normalize(sem_embeddings, p=2, dim=1)
        sim_mat  = torch.mm(sem_norm, sem_norm.t())
        sim_mat.fill_diagonal_(0)

        pos_mask = (adj > 0) & (sim_mat > feature_sim_threshold)
        pos_mask.fill_diagonal_(False)

        neg_mask = (adj > 0) & (sim_mat < neg_feature_threshold) & (~pos_mask)
        neg_mask.fill_diagonal_(False)

        pos_nz = pos_mask.nonzero(as_tuple=False)
        neg_nz = neg_mask.nonzero(as_tuple=False)

        pos_pairs = {}
        neg_pairs = {}

        for i in range(batch_size):
            pos_idx = pos_nz[pos_nz[:, 0] == i, 1].cpu().numpy().tolist()
            pos_pairs[i] = [i] + pos_idx

            neg_idx = neg_nz[neg_nz[:, 0] == i, 1].cpu().numpy().tolist()

            if len(neg_idx) < 5:
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                mask[pos_pairs[i]] = False
                mask[neg_idx] = False
                remaining = torch.arange(batch_size, device=device)[mask]
                if len(remaining) > 0:
                    extra = remaining[
                        torch.randperm(len(remaining), device=device)[:min(10, len(remaining))]
                    ].cpu().numpy().tolist()
                    neg_idx.extend(extra)

            neg_pairs[i] = neg_idx

        return pos_pairs, neg_pairs

    def build_topology_graph(self, data):
        drug_idx, prot_idx = data[:, 0], data[:, 1]
        num_nodes = len(data)
        edge_list = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if drug_idx[i] == drug_idx[j] or prot_idx[i] == prot_idx[j]:
                    edge_list.extend([[i, j], [j, i]])

        if edge_list:
            edge = torch.tensor(edge_list, dtype=torch.long).t().to(data.device)
        elif num_nodes > 1:
            edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().to(data.device)
        else:
            edge = torch.empty((2, 0), dtype=torch.long, device=data.device)

        return edge

    def build_semantic_graph(self, DPP_emb, k=5):
        num_nodes = DPP_emb.size(0)
        k_eff = min(k, num_nodes - 1)

        if k_eff <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=DPP_emb.device)

        norm = F.normalize(DPP_emb, p=2, dim=1)
        sim  = torch.mm(norm, norm.t())
        sim.fill_diagonal_(-float('inf'))

        _, topk = torch.topk(sim, k=k_eff, dim=1, largest=True)

        src_list, dst_list = [], []
        for i in range(num_nodes):
            for j in topk[i].tolist():
                src_list.extend([i, j])
                dst_list.extend([j, i])

        if not src_list:
            return torch.empty((2, 0), dtype=torch.long, device=DPP_emb.device)

        edge = torch.tensor([src_list, dst_list], dtype=torch.long, device=DPP_emb.device)
        return torch.unique(edge, dim=1)

    def _build_dgl_graph(self, edge, num_nodes):
        if edge.size(1) > 0:
            graph = dgl.graph(
                (edge[0].long(), edge[1].long()),
                num_nodes=num_nodes
            )
        else:
            empty = torch.empty((0,), dtype=torch.long, device=edge.device)
            graph = dgl.graph((empty, empty), num_nodes=num_nodes)

        return dgl.add_self_loop(graph)