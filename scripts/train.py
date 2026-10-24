"""Train the ModFormer read-level binary classification model.

Function: isolate training, validation, and test data by chromosome, then use
PyTorch Lightning for mixed-precision training, validation metric computation,
early stopping, and checkpointing.
Inputs: v6 PT shards; each batch contains eight fields, while the model uses
only read/reference 31-mers, signals [B,31,16], stats [B,31,5], and labels [B].
Outputs: Lightning checkpoints, CSV logs, and validation classification metrics.
Key constraints: the default split is train=chr2-chr21/chrX/chrY,
validation=chr22, and test=chr1; test data are not used for training or
checkpoint selection.
"""

import argparse
import contextlib
import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, MatthewsCorrCoef, Precision, Recall

from src.modformer.dataset import create_chr_level_data_loaders
from src.modformer.model import ModFormer

os.environ["NCCL_P2P_DISABLE"] = "1"
torch.set_float32_matmul_precision("high")

DEFAULT_DATASET_DIR = "data/processed"
DEFAULT_LOG_DIR = "outputs/training"
DEFAULT_NUM_GPUS = 1

SEQ_LEN = 31
SIGNAL_LEN = 16
STATS_DIM = 5
CLASS_NUM = 2
HIDDEN_SIZE = 1024
N_VOCAB = 5
N_EMBED = 256
PER_GPU_BATCH_SIZE = 2048

resume_from_checkpoint = ""
checkpoint_path = ""

def is_global_zero() -> bool:
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))) == 0

def rank_zero_print(*args, **kwargs):
    if is_global_zero():
        print(*args, **kwargs)

@contextlib.contextmanager
def suppress_nonzero_rank_output():
    if is_global_zero():
        yield
        return
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield

def parse_chromosome_list(spec: str) -> list[str]:
    chroms = []
    for part in spec.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            start, end = [x.strip().removeprefix('chr') for x in part.split('-', 1)]
            if not (start.isdigit() and end.isdigit()): raise ValueError(f"Unsupported chromosome range: {part}")
            chroms.extend([f"chr{i}" for i in range(int(start), int(end) + 1)])
        else:
            chroms.append(part if part.startswith('chr') else f'chr{part}')
    return chroms

def parse_train_args():
    p = argparse.ArgumentParser(description='Train ModFormer with chromosome-level splits')
    p.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR)
    p.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR)
    p.add_argument('--num-gpus', type=int, default=DEFAULT_NUM_GPUS)
    p.add_argument('--gpu-ids', type=str, default=None)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=PER_GPU_BATCH_SIZE)
    p.add_argument('--limit_val_batches', type=float, default=1.0)
    p.add_argument('--file-fraction', type=float, default=1.0)
    p.add_argument('--rebuild-shards', action='store_true')
    p.add_argument('--train-chroms', type=str, default='chr2-chr21,chrX,chrY')
    p.add_argument('--val-chroms', type=str, default='chr22')
    p.add_argument('--test-chroms', type=str, default='chr1')
    return p.parse_args()

def resolve_gpu_devices(num_gpus: int):
    if num_gpus == 1: return [0], 'auto'
    if num_gpus == 2: return [0, 1], 'ddp'
    if num_gpus == 3: return [0, 1, 2], 'ddp'
    if num_gpus == 4: return [0, 1, 2, 3], 'ddp'
    if num_gpus < 1: raise ValueError('num_gpus must be >= 1')
    return list(range(num_gpus)), 'ddp'

def compute_lr_and_warmup(_effective_batch_size: int):
    if checkpoint_path: return 2e-5, 0.05, True
    return 5e-5, 0.10, False

class LightningModel(pl.LightningModule):
    """Wrap ModFormer classification loss, global validation metrics, and optimizer scheduling."""

    def __init__(self, lr, warmup_ratio=0.10, class_weights=None):
        super().__init__()
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = ModFormer(seq_len=SEQ_LEN, signal_len=SIGNAL_LEN, stats_dim=STATS_DIM, num_classes=CLASS_NUM, hidden_size=HIDDEN_SIZE, vocab_size=N_VOCAB, embedding_size=N_EMBED, use_checkpoint=False)
        self.weight_rank = class_weights.float() if class_weights is not None else torch.tensor([1.0, 1.0])
        rank_zero_print(f"Class weights: {self.weight_rank}")
        self.criterion = nn.CrossEntropyLoss(weight=self.weight_rank, label_smoothing=0.05)
        self.train_accuracy = Accuracy(task='binary')
        self.val_metrics = self._make_metrics()
    @staticmethod
    def _make_metrics():
        return nn.ModuleDict({
            'accuracy': Accuracy(task='binary'),
            'precision': Precision(task='binary', zero_division=0),
            'recall': Recall(task='binary', zero_division=0),
            'f1': F1Score(task='binary', zero_division=0),
            'mcc': MatthewsCorrCoef(task='binary'),
            'auroc': AUROC(task='binary'),
            'ap': AveragePrecision(task='binary'),
        })
    def load_model_weights(self, path):
        ckpt = torch.load(path, map_location='cpu')
        raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        state = {k.replace('model.', ''): v for k, v in raw.items() if not k.startswith('criterion')}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        rank_zero_print(f"Weights loaded: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    @staticmethod
    def unpack_batch(batch):
        if len(batch) != 8: raise ValueError(f"v6 batch must contain 8 items, got {len(batch)}")
        read_kmer, ref_kmer, signals, stats, labels, _pos, _chr_code, _strand = batch
        return read_kmer.long(), ref_kmer.long(), signals.float(), stats.float(), labels.long()
    def _forward_loss(self, batch):
        read_kmer, ref_kmer, signals, stats, labels = self.unpack_batch(batch)
        logits = self.model(read_kmer, ref_kmer, stats, signals)
        loss = self.criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        return loss, preds, probs, labels
    @staticmethod
    def _update_metrics(metrics, preds, probs, labels):
        for name, metric in metrics.items():
            if name in {'auroc', 'ap'}: metric(probs, labels)
            else: metric(preds, labels)
    def training_step(self, batch, batch_idx):
        loss, preds, _, labels = self._forward_loss(batch)
        if torch.isnan(loss) or torch.isinf(loss): raise RuntimeError(f"Batch {batch_idx} produced a non-finite loss: {loss}")
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, preds, probs, labels = self._forward_loss(batch)
        self._update_metrics(self.val_metrics, preds, probs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def _compute_log_reset(self, metrics, prefix, title):
        values = {name: metric.compute() for name, metric in metrics.items()}
        for name, value in values.items():
            self.log(f'{prefix}_{name}', value, prog_bar=name in {'f1', 'auroc'}, sync_dist=True, add_dataloader_idx=False)
        rank_zero_print(f"\nEpoch {self.current_epoch} {title}:")
        for name in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'ap']:
            rank_zero_print(f"  {name:9s}: {values[name]:.4f}")
        rank_zero_print()
        for metric in metrics.values(): metric.reset()
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            for metric in self.val_metrics.values(): metric.reset()
            return
        self._compute_log_reset(self.val_metrics, 'val', 'chr22 validation metrics')
    def on_train_epoch_end(self):
        self.train_accuracy.reset()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        rank_zero_print(f"\nLR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, lr={self.lr:.6f}\n")
        def lr_lambda(step):
            if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), 'interval': 'step'}}

def main():
    args = parse_train_args()
    pl.seed_everything(args.seed, workers=True)
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        strategy = 'auto' if len(gpu_ids) == 1 else 'ddp'
    else:
        gpu_ids, strategy = resolve_gpu_devices(args.num_gpus)
    lr, warmup_ratio, finetune = compute_lr_and_warmup(args.batch_size * len(gpu_ids))
    train_chroms = parse_chromosome_list(args.train_chroms)
    val_chroms = parse_chromosome_list(args.val_chroms)
    test_chroms = parse_chromosome_list(args.test_chroms)
    rank_zero_print('=' * 70)
    rank_zero_print(f"Modformer v6: train={train_chroms}, val={val_chroms}, test={test_chroms}")
    rank_zero_print(f"lr={lr}, warmup={warmup_ratio}, finetune={finetune}, gpus={gpu_ids}, strategy={strategy}")
    rank_zero_print('=' * 70)
    with suppress_nonzero_rank_output():
        train_loader, valid_loader, _test_loader, class_weights = create_chr_level_data_loaders(
            data_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_seq_len=SEQ_LEN,
            random_seed=args.seed,
            file_fraction=args.file_fraction,
            train_chromosomes=train_chroms,
            val_chromosomes=val_chroms,
            test_chromosomes=test_chroms,
            rebuild_shards=args.rebuild_shards,
        )
    model = LightningModel(lr=lr, warmup_ratio=warmup_ratio, class_weights=class_weights)
    if checkpoint_path: model.load_model_weights(checkpoint_path)
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt = pl.callbacks.ModelCheckpoint(monitor='val_f1', filename='modformer-{epoch:02d}-{val_f1:.4f}', save_top_k=50, mode='max', save_last=True)
    callbacks = [ckpt, pl.callbacks.EarlyStopping(monitor='val_f1', patience=10, mode='max', min_delta=0.001, verbose=True), LearningRateMonitor(logging_interval='epoch', log_momentum=False)]
    trainer = pl.Trainer(default_root_dir=args.log_dir, devices=gpu_ids, max_epochs=args.max_epochs, accelerator='gpu', strategy=strategy, logger=CSVLogger(save_dir=args.log_dir, name='csv_logs'), precision='16-mixed', callbacks=callbacks, gradient_clip_val=1.0, gradient_clip_algorithm='norm', deterministic=False, benchmark=True, log_every_n_steps=50, limit_val_batches=args.limit_val_batches)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=resume_from_checkpoint or None)
    rank_zero_print(f"Best model: {ckpt.best_model_path}")
    rank_zero_print(f"Best chr22 val_f1: {ckpt.best_model_score:.4f}")

if __name__ == '__main__':
    main()
