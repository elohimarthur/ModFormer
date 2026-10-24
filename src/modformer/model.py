"""Define the multimodal ModFormer CpG modification classifier.

Function: fuse read/reference 31-mers, per-base raw signals, and signal
statistics to produce read-level binary-classification logits.
Inputs: read_kmer and ref_kmer are [B, S], signals are [B, S, L], and
signal_stats are [B, S, F]. Defaults are S=31, L=16, and F=5.
Output: [B, num_classes] logits; the loss function should consume logits directly.
Key constraints: the default base-encoding vocabulary size is 5, and
embedding_size must be divisible by eight attention heads.
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Dict, Any
from torch.utils.checkpoint import checkpoint

class SignalWaveformEncoder(nn.Module):
    """Encode per-base waveforms from ``[B, S, L]`` to ``[B, S, E]``."""

    def __init__(self, signal_len: int, embedding_size: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(128, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        b, s, l = signals.shape
        x = signals.reshape(-1, 1, l)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.projection(x)
        return x.view(b, s, -1)

class SimplifiedGatedFusion(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.LayerNorm(embedding_size))
    def forward(self, seq_embedded: torch.Tensor, signal_stats_embedded: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([seq_embedded, signal_stats_embedded], dim=-1)
        gate = self.gate(combined)
        fused = self.fusion(combined)
        residual = 0.5 * (seq_embedded + signal_stats_embedded)
        return gate * fused + (1 - gate) * residual

class SimplifiedAttentionPooling(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.attention = nn.Linear(embedding_size, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attention(x), dim=1)
        return (x * w).sum(dim=1)

class SymmetricCrossAttention(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.seq_query_attn = nn.MultiheadAttention(embedding_size, num_heads=8, dropout=0.1, batch_first=True)
        self.sig_query_attn = nn.MultiheadAttention(embedding_size, num_heads=8, dropout=0.1, batch_first=True)
        self.fusion = nn.Linear(embedding_size * 2, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_size * 2, embedding_size),
        )
        self.norm2 = nn.LayerNorm(embedding_size)
    def forward(self, seq_features: torch.Tensor, sig_features: torch.Tensor) -> torch.Tensor:
        seq_attended, _ = self.seq_query_attn(seq_features, sig_features, sig_features)
        sig_attended, _ = self.sig_query_attn(sig_features, seq_features, seq_features)
        fused = self.fusion(torch.cat([seq_attended, sig_attended], dim=-1))
        x = self.norm1(fused + 0.5 * seq_features + 0.5 * sig_features)
        return self.norm2(self.ffn(x) + x)

class ModFormer(nn.Module):
    """Classify reads by fusing sequence, waveform, and signal-statistic features.

    ``seq_len``, ``signal_len``, and ``stats_dim`` must match the data-generation
    schema. ``forward`` returns logits without applying softmax.
    """

    def __init__(self, seq_len: int = 31, signal_len: int = 16, stats_dim: int = 5, num_classes: int = 2,
                 embedding_size: int = 256, hidden_size: int = 1024, vocab_size: int = 5,
                 num_cross_layers: int = 2, use_checkpoint: bool = False):
        super().__init__()
        assert embedding_size % 8 == 0
        self.seq_len = seq_len
        self.signal_len = signal_len
        self.stats_dim = stats_dim
        self.embedding_size = embedding_size
        self.use_checkpoint = use_checkpoint
        self.read_embed = nn.Embedding(vocab_size, embedding_size)
        self.ref_embed = nn.Embedding(vocab_size, embedding_size)
        self.seq_input_proj = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.signal_stats_norm = nn.LayerNorm(stats_dim)
        self.signal_stats_linear = nn.Sequential(
            nn.Linear(stats_dim, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
        )
        self.sequence_fusion = SimplifiedGatedFusion(embedding_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embedding_size) * 0.02)
        self.seq_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8, dim_feedforward=hidden_size, dropout=0.1, batch_first=True, norm_first=True),
            num_layers=4,
            enable_nested_tensor=False,
        )
        self.signal_waveform_encoder = SignalWaveformEncoder(signal_len, embedding_size)
        self.signal_transformer = torchaudio.models.Conformer(
            input_dim=embedding_size,
            num_heads=8,
            ffn_dim=hidden_size,
            num_layers=4,
            depthwise_conv_kernel_size=7,
            dropout=0.1,
        )
        self.cross_modal_layers = nn.ModuleList([SymmetricCrossAttention(embedding_size) for _ in range(num_cross_layers)])
        self.attention_pool = SimplifiedAttentionPooling(embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, read_kmer: torch.Tensor, ref_kmer: torch.Tensor, signal_stats: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        assert read_kmer.size(1) == self.seq_len
        assert ref_kmer.size(1) == self.seq_len
        assert signals.size(1) == self.seq_len and signals.size(-1) == self.signal_len
        assert signal_stats.size(1) == self.seq_len and signal_stats.size(-1) == self.stats_dim
        read_emb = self.read_embed(read_kmer)
        ref_emb = self.ref_embed(ref_kmer)
        seq_emb = self.seq_input_proj(torch.cat([read_emb, ref_emb], dim=-1))
        stats_emb = self.signal_stats_linear(self.signal_stats_norm(torch.clamp(signal_stats, min=-100, max=100)))
        seq_features = self.sequence_fusion(seq_emb, stats_emb) + self.pos_embed
        if self.use_checkpoint and self.training:
            out_seq = checkpoint(self.seq_transformer, seq_features, use_reentrant=False)
        else:
            out_seq = self.seq_transformer(seq_features)
        sig_features = self.signal_waveform_encoder(signals) + self.pos_embed
        lengths = torch.full((signals.size(0),), sig_features.size(1), device=signals.device, dtype=torch.long)
        if self.use_checkpoint and self.training:
            def conformer_forward(x, lens):
                y, _ = self.signal_transformer(x, lens)
                return y
            out_sig = checkpoint(conformer_forward, sig_features, lengths, use_reentrant=False)
        else:
            out_sig, _ = self.signal_transformer(sig_features, lengths)
        fused = out_seq
        for layer in self.cross_modal_layers:
            fused = layer(fused, out_sig)
        return self.classifier(self.attention_pool(fused))
    def get_model_size(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_params': total, 'trainable_params': trainable, 'model_size_mb': total * 4 / 1024 / 1024}

if __name__ == '__main__':
    model = ModFormer()
    b = 2
    out = model(torch.randint(0,5,(b,31)), torch.randint(0,5,(b,31)), torch.randn(b,31,5), torch.randn(b,31,16))
    print(out.shape)
    print(model.get_model_size())
