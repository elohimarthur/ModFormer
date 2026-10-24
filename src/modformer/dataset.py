"""Load chromosome-isolated ModFormer PT shards.

Function: build chromosome-specific training, validation, and test shard views,
then stream samples through an IterableDataset with file assignment across DDP
ranks and DataLoader workers.
Inputs: each ``.pt`` file contains a TensorDataset with tensors ordered as
read_kmer [N,31], ref_kmer [N,31], signals [N,31,16], stats [N,31,5], labels [N],
positions [N], chr_codes [N], and strands [N].
Outputs: training, validation, and test DataLoaders, plus class weights computed
from the training-label distribution.
Key constraints: the default split is train=chr2-chr21/chrX/chrY,
validation=chr22, and test=chr1; positions, chr_codes, and strands are returned
only as metadata and are not model inputs.
"""

import gc
import glob
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from tqdm import tqdm

DEFAULT_TRAIN_CHROMS = [f"chr{i}" for i in range(2, 22)] + ["chrX", "chrY"]
DEFAULT_VAL_CHROMS = ["chr22"]
DEFAULT_TEST_CHROMS = ["chr1"]
DEFAULT_TRAIN_SUBDIR = "train_shards"
DEFAULT_VAL_SUBDIR = "validation_shards"
DEFAULT_TEST_SUBDIR = "test_shards"
EXPECTED_SIGNAL_LEN = 16

def _load_shard(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if 'mmap' in str(e).lower():
            return torch.load(path, map_location='cpu', weights_only=False)
        raise

class _EmptyIterable(IterableDataset):
    def __iter__(self): yield from ()
    def __len__(self): return 0

def _has_pt_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.pt"))

def _normalize_chroms(chromosomes: list[str]) -> list[str]:
    out=[]
    for c in chromosomes:
        c=str(c).strip()
        if c: out.append(c if c.startswith('chr') else f'chr{c}')
    return out

def _collect_chromosome_files(root: Path, chromosomes: list[str]) -> list[Path]:
    files=[]; missing=[]; empty=[]
    for chrom in chromosomes:
        d=root/chrom
        if not d.exists(): missing.append(chrom); continue
        fs=sorted(d.glob('*.pt'))
        if not fs: empty.append(chrom); continue
        files.extend(fs)
    if missing: print(f"[v6 chr-split] Warning: missing chromosome directories: {missing}")
    if empty: print(f"[v6 chr-split] Warning: chromosome directories contain no .pt files: {empty}")
    return files

def _clear_links(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob('*.pt'): f.unlink()

def _link_or_copy(files: list[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst=dst_dir/f"{src.parent.name}__{src.name}"
        if dst.exists():
            h=hashlib.sha1(str(src).encode()).hexdigest()[:10]
            dst=dst_dir/f"{src.parent.name}__{src.stem}__{h}{src.suffix}"
        if dst.exists(): continue
        try: os.symlink(src,dst)
        except OSError:
            try: os.link(src,dst)
            except OSError:
                import shutil; shutil.copy2(src,dst)

def _manifest_path(root: Path, train_subdir: str, val_subdir: str, test_subdir: str) -> Path:
    h=hashlib.sha1(f"v6__{train_subdir}__{val_subdir}__{test_subdir}".encode()).hexdigest()[:10]
    return root/f".chr_split_manifest_v6_{h}.json"

def ensure_chr_level_shards(data_dir: str, train_shard_subdir=DEFAULT_TRAIN_SUBDIR, val_shard_subdir=DEFAULT_VAL_SUBDIR, test_shard_subdir=DEFAULT_TEST_SUBDIR, train_chromosomes=None, val_chromosomes=None, test_chromosomes=None, rebuild=False):
    root=Path(data_dir)
    if not root.exists(): raise RuntimeError(f"Data directory does not exist: {root}")
    train_chromosomes=_normalize_chroms(train_chromosomes or DEFAULT_TRAIN_CHROMS)
    val_chromosomes=_normalize_chroms(val_chromosomes or DEFAULT_VAL_CHROMS)
    test_chromosomes=_normalize_chroms(test_chromosomes or DEFAULT_TEST_CHROMS)
    train_dir=root/train_shard_subdir; val_dir=root/val_shard_subdir; test_dir=root/test_shard_subdir
    manifest=_manifest_path(root,train_shard_subdir,val_shard_subdir,test_shard_subdir)
    valid=False
    if manifest.exists() and not rebuild and _has_pt_files(train_dir) and _has_pt_files(val_dir):
        try:
            m=json.loads(manifest.read_text())
            valid=(m.get('version')==6 and m.get('train_chromosomes')==train_chromosomes and m.get('val_chromosomes')==val_chromosomes and m.get('test_chromosomes')==test_chromosomes and m.get('train_file_count')==len(list(train_dir.glob('*.pt'))) and m.get('val_file_count')==len(list(val_dir.glob('*.pt'))) and m.get('test_file_count')==len(list(test_dir.glob('*.pt'))))
        except Exception: valid=False
    if valid:
        print(f"[v6 chr-split] Existing shards detected; skipping build: {manifest}")
        return train_dir,val_dir,test_dir
    train_files=_collect_chromosome_files(root,train_chromosomes); val_files=_collect_chromosome_files(root,val_chromosomes); test_files=_collect_chromosome_files(root,test_chromosomes)
    if not train_files: raise RuntimeError(f"No training .pt files found: {train_chromosomes}")
    if not val_files: raise RuntimeError(f"No validation .pt files found: {val_chromosomes}")
    if not test_files: print(f"[v6 chr-split] Warning: no test .pt files found: {test_chromosomes}")
    for d in (train_dir,val_dir,test_dir): _clear_links(d)
    _link_or_copy(train_files,train_dir); _link_or_copy(val_files,val_dir); _link_or_copy(test_files,test_dir)
    payload={'version':6,'mode':'chromosome_level_split_v6','train_chromosomes':train_chromosomes,'val_chromosomes':val_chromosomes,'test_chromosomes':test_chromosomes,'train_file_count':len(train_files),'val_file_count':len(val_files),'test_file_count':len(test_files)}
    manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(f"[v6 chr-split] train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    return train_dir,val_dir,test_dir

class ChromosomeShardDataset(IterableDataset):
    """Yield individual v6 samples from shards and distribute work across ranks and workers.

    ``target_seq_len`` only performs odd-length cropping centered on the CpG;
    labels and position, chromosome, and strand metadata remain unchanged.
    """

    def __init__(self, shard_dir: str, split_tag='train', shuffle=True, target_seq_len: Optional[int]=None, random_seed=42, file_fraction=1.0):
        super().__init__(); self.shard_dir=Path(shard_dir); self.split_tag=split_tag; self.shuffle=shuffle; self.target_seq_len=target_seq_len; self.original_seq_len=31; self.random_seed=random_seed
        if target_seq_len is not None:
            assert target_seq_len%2==1 and target_seq_len<=self.original_seq_len
        all_files=sorted(glob.glob(str(self.shard_dir/'*.pt')))
        self.data_files=[]
        skipped_empty=0
        for fp in all_files:
            try:
                if os.path.getsize(fp) <= 0:
                    skipped_empty += 1
                    continue
            except OSError:
                skipped_empty += 1
                continue
            self.data_files.append(fp)
        if skipped_empty:
            print(f"[v6:{split_tag}] Skipped empty/inaccessible shards: {skipped_empty}")
        if file_fraction<1.0 and self.data_files:
            self.data_files=self.data_files[:max(1,int(len(self.data_files)*file_fraction))]
        if not self.data_files: raise RuntimeError(f"[v6:{split_tag}] No usable shards in directory: {self.shard_dir}")
        print(f"[v6:{split_tag}] Shard directory {self.shard_dir}, {len(self.data_files)} usable .pt files")
        self._load_metadata(); self.class_weights=self._compute_class_weights()
    def _cache_path(self):
        d=self.shard_dir/'.cache'; d.mkdir(parents=True, exist_ok=True); return d/f"shard_metadata_v6_signal{EXPECTED_SIGNAL_LEN}_{self.split_tag}.pt"
    def _load_metadata(self):
        cache=self._cache_path(); meta={}
        if cache.exists():
            try: meta=torch.load(cache,weights_only=False); print(f"  Loaded v6 shard metadata cache: {len(meta)} entries")
            except Exception: meta={}
        missing=[p for p in self.data_files if p not in meta]
        if missing:
            for fp in tqdm(missing,desc='Scanning v6 shards',leave=False):
                try:
                    data=_load_shard(fp)
                    if isinstance(data,TensorDataset):
                        if len(data.tensors)!=8:
                            raise ValueError(f'Not a v6 TensorDataset: expected 8 tensors, got {len(data.tensors)}')
                        if data.tensors[2].shape[-1]!=EXPECTED_SIGNAL_LEN:
                            raise ValueError(f'Signal length mismatch: expected {EXPECTED_SIGNAL_LEN}, got {data.tensors[2].shape[-1]}; regenerate the .pt data')
                        labels=data.tensors[4]
                    else:
                        first=data[0] if len(data)>0 else None
                        if first is not None and len(first)!=8:
                            raise ValueError(f'Not a v6 data item: expected 8 fields, got {len(first)}')
                        if first is not None and first[2].shape[-1]!=EXPECTED_SIGNAL_LEN:
                            raise ValueError(f'Signal length mismatch: expected {EXPECTED_SIGNAL_LEN}, got {first[2].shape[-1]}; regenerate the .pt data')
                        labels=torch.tensor([x[4] for x in data])
                    meta[fp]={'sample_count':len(data),'label_0':int((labels==0).sum()),'label_1':int((labels==1).sum()),'valid':True}
                    del data; gc.collect()
                except Exception as e:
                    print(f"  Scan failed; skipping {fp}: {e}"); meta[fp]={'sample_count':0,'label_0':0,'label_1':0,'valid':False}
            torch.save(meta,cache)
        valid_files=[]
        self.total_samples=0; self.label_distribution={0:0,1:0}
        invalid_count=0
        for fp in self.data_files:
            m=meta.get(fp,{'sample_count':0,'label_0':0,'label_1':0,'valid':False})
            if not m.get('valid', m.get('sample_count',0)>0) or m.get('sample_count',0)<=0:
                invalid_count += 1
                continue
            valid_files.append(fp)
            self.total_samples+=m['sample_count']; self.label_distribution[0]+=m['label_0']; self.label_distribution[1]+=m['label_1']
        self.data_files=valid_files
        if invalid_count:
            print(f"  Removed invalid shards from this training run: {invalid_count}")
        if not self.data_files or self.total_samples<=0:
            raise RuntimeError(f"[v6:{self.split_tag}] No trainable samples; check whether shards are empty or corrupted: {self.shard_dir}")
        tot=sum(self.label_distribution.values())
        if tot: print(f"  Label distribution - 0: {self.label_distribution[0]/tot:.2%}, 1: {self.label_distribution[1]/tot:.2%}")
        print(f"  Statistics complete: {self.total_samples:,} samples")
    def _compute_class_weights(self):
        if self.label_distribution[0]==0 or self.label_distribution[1]==0: return torch.FloatTensor([1.0,1.0])
        y=[0]*self.label_distribution[0]+[1]*self.label_distribution[1]
        return torch.FloatTensor(compute_class_weight('balanced',classes=np.unique(y),y=y))
    def _center_crop(self,item):
        if self.target_seq_len is None or self.target_seq_len==self.original_seq_len: return item
        crop=(self.original_seq_len-self.target_seq_len)//2; st,en=crop,self.original_seq_len-crop
        read_kmer,ref_kmer,signals,stats,labels,pos,chr_code,strand=item
        return read_kmer[st:en],ref_kmer[st:en],signals[st:en,:],stats[st:en,:],labels,pos,chr_code,strand
    def __iter__(self):
        if not hasattr(self,'_epoch_iter_count'): self._epoch_iter_count=0
        epoch=self._epoch_iter_count; self._epoch_iter_count+=1
        files=list(self.data_files); rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0; world=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if self.shuffle:
            g=torch.Generator(); g.manual_seed(int(epoch)+int(self.random_seed)); perm=torch.randperm(len(files),generator=g).tolist(); files=[files[i] for i in perm]
        rank_files=files[rank::world] or ([files[rank%len(files)]] if files else [])
        wi=torch.utils.data.get_worker_info(); nw=wi.num_workers if wi else 1; wid=wi.id if wi else 0
        quota=(self.total_samples//max(1,world))//nw + (1 if wid < ((self.total_samples//max(1,world))%nw) else 0)
        rng=random.Random(int(epoch)*1000003+int(self.random_seed)+rank*7919+wid*17); yielded=0; pass_id=0
        while yielded<quota:
            order=list(rank_files)
            if self.shuffle:
                random.Random(int(epoch)*1000003+int(self.random_seed)+rank*7919+pass_id*10007).shuffle(order)
            for fp in order:
                if yielded>=quota: break
                try:
                    data=_load_shard(fp)
                    if isinstance(data,TensorDataset) and len(data.tensors)!=8:
                        raise ValueError(f'Not a v6 TensorDataset: expected 8 tensors, got {len(data.tensors)}')
                    idxs=list(range(wid,len(data),nw))
                    if self.shuffle: rng.shuffle(idxs)
                    for idx in idxs:
                        if yielded>=quota: break
                        yield self._center_crop(data[idx]); yielded+=1
                    del data; gc.collect()
                except Exception as e: print(f"[v6] Failed to load {fp}: {e}")
            pass_id+=1
            if pass_id>100000: raise RuntimeError(f"v6 could not satisfy worker_quota={quota}")
    def __len__(self):
        world=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return self.total_samples//world

def create_chr_level_data_loaders(data_dir: str, train_shard_subdir=DEFAULT_TRAIN_SUBDIR, val_shard_subdir=DEFAULT_VAL_SUBDIR, test_shard_subdir=DEFAULT_TEST_SUBDIR, batch_size=32, num_workers=4, target_seq_len=None, random_seed=42, file_fraction=1.0, train_chromosomes=None, val_chromosomes=None, test_chromosomes=None, rebuild_shards=False):
    """Build chromosome-isolated training, validation, and test DataLoaders.

    Returns ``(train_loader, val_loader, test_loader, class_weights)``. Class
    weights are computed only from the training-shard label distribution; test
    data do not contribute to this calculation.
    """
    train_chromosomes=_normalize_chroms(train_chromosomes or DEFAULT_TRAIN_CHROMS); val_chromosomes=_normalize_chroms(val_chromosomes or DEFAULT_VAL_CHROMS); test_chromosomes=_normalize_chroms(test_chromosomes or DEFAULT_TEST_CHROMS)
    train_dir,val_dir,test_dir=ensure_chr_level_shards(data_dir,train_shard_subdir,val_shard_subdir,test_shard_subdir,train_chromosomes,val_chromosomes,test_chromosomes,rebuild_shards)
    print('='*70); print('v6 chromosome-level dataloader configuration'); print(f'data_dir: {data_dir}'); print(f'train: {train_chromosomes}'); print(f'val: {val_chromosomes}'); print(f'test: {test_chromosomes}'); print('='*70)
    train_ds=ChromosomeShardDataset(str(train_dir),'train_v6',True,target_seq_len,random_seed,file_fraction)
    val_ds=ChromosomeShardDataset(str(val_dir),'val_v6',False,target_seq_len,random_seed,file_fraction)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True,persistent_workers=bool(num_workers))
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=max(1,num_workers),pin_memory=True,persistent_workers=bool(num_workers))
    if test_dir.exists() and any(test_dir.glob('*.pt')):
        test_ds=ChromosomeShardDataset(str(test_dir),'test_v6',False,target_seq_len,random_seed,file_fraction)
        test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=max(1,min(num_workers,4)),pin_memory=True,persistent_workers=bool(num_workers))
        print(f"  Test samples (global): {test_ds.total_samples:,} | per rank ~= {len(test_ds):,}")
    else:
        test_loader=DataLoader(_EmptyIterable(),batch_size=batch_size,num_workers=0); print('  Warning: test_loader is empty')
    print(f"  Training samples (global): {train_ds.total_samples:,} | per rank ~= {len(train_ds):,}"); print(f"  Validation samples (global): {val_ds.total_samples:,} | per rank ~= {len(val_ds):,}"); print(f"  Class weights: {train_ds.class_weights.tolist()}")
    return train_loader,val_loader,test_loader,train_ds.class_weights

if __name__=='__main__': print('dataloader_v6.py')
