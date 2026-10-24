#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ModFormer PT shards from BAM, raw HDF5 signals, and ground truth.

Function: align reads, references, and raw signals using BAM alignments and the
``mv`` tag; extract 31-nt windows around labeled CpGs; and incrementally write
TensorDatasets by chromosome.
Inputs: BAM, HDF5, reference FASTA, reference-to-chromosome mapping, signal
configuration, and categorical-label or continuous-frequency pickle files.
Genomic coordinates are 0-based.
Output tensor order: read_kmer [N,31], ref_kmer [N,31], signals [N,31,16],
stats [N,31,5], labels [N], positions [N], chr_codes [N], strands [N].
Key constraints: forward- and reverse-strand windows are oriented around the
central C; ``mv`` values are multiplied by stride and interpreted as raw-signal
indices; output directories containing existing PT shards are not silently reused.
"""
import os, h5py, pysam, numpy as np, array, json, torch, pickle, time, logging
from pathlib import Path
from torch.utils.data import TensorDataset
from scipy.stats import median_abs_deviation
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict

BASE2CODE={'A':0,'C':1,'G':2,'T':3,'N':4}
COMPLEMENT={'A':'T','T':'A','C':'G','G':'C','N':'N'}
CHR_TO_CODE={**{f'chr{i}':i for i in range(1,23)},'chrX':23,'chrY':24}
FRAGMENT_LENGTH=31
CONTEXT_BASES=15
DEFAULT_SIGNAL_LEN=16
CLIP_VALUE=5.0
MIN_BLOCKS_LEN_FACTOR=0.1
MAX_BLOCKS_LEN_FACTOR=8.0
_G_METHYLATION_DATA={}
_G_REF_TO_CHR={}
_G_CONFIG={}
_G_LABEL_MODE='categorical'
_G_GT_POSITIONS={}
_W_REF_FASTA=None

def _worker_init(ref_fasta_path,label_mode):
    global _W_REF_FASTA,_G_LABEL_MODE
    _G_LABEL_MODE=label_mode
    if ref_fasta_path and os.path.exists(ref_fasta_path):
        _W_REF_FASTA=pysam.FastaFile(ref_fasta_path)

def reverse_complement(seq):
    return ''.join(COMPLEMENT.get(b,b) for b in reversed(seq))

def convert_int_signal_to_physical(signal_x):
    x=np.asarray(signal_x,dtype=np.float32)
    return (x*2.5/6144.0)*(-200.0/11.0)/4.0/2.5

def normalize_signal(signal_raw,scaling,offset):
    scaled=np.asarray(scaling*(signal_raw+offset),dtype=np.float32)
    med=np.median(scaled); mad=median_abs_deviation(scaled)
    norm=scaled-med if mad==0 else (scaled-med)/mad
    return np.clip(norm,-CLIP_VALUE,CLIP_VALUE).astype(np.float32)

def encode_bases(seq):
    return [BASE2CODE.get(b.upper(),4) for b in seq]

def calculate_sequence_similarity(seq1,seq2):
    if not seq1 or not seq2 or len(seq1)!=len(seq2): return 0.0
    return 100.0*sum(a.upper()==b.upper() for a,b in zip(seq1,seq2))/len(seq1)

def convert_move_table(move_table,stride):
    return np.asarray(move_table,dtype=np.int64)*int(stride)

def pad_or_truncate_base_signals(signals_list,signal_len=DEFAULT_SIGNAL_LEN):
    out=[]
    for s in signals_list:
        arr=np.asarray(s,dtype=np.float32)
        if arr.size<signal_len:
            x=np.zeros(signal_len,dtype=np.float32)
            start=(signal_len-arr.size)//2
            if arr.size: x[start:start+arr.size]=arr
        elif arr.size>signal_len:
            start=(arr.size-signal_len)//2
            x=arr[start:start+signal_len]
        else:
            x=arr
        out.append(x.tolist())
    return out

def compute_base_signal_stats(signals_list):
    out=[]
    for s in signals_list:
        arr=np.asarray(s,dtype=np.float32); n=arr.size
        if n==0:
            out.append([0.0,0.0,0.0,0.0,0.0]); continue
        med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
        out.append([float(np.mean(arr)),float(np.std(arr)),med,mad,float(np.log10(n+1.0))])
    return out

def build_query_events(signal,move_table,seq_len):
    if len(move_table)!=seq_len:
        raise ValueError(f'mv length {len(move_table)} != sequence length {seq_len}')
    if seq_len and move_table[0]<0:
        raise ValueError('mv start must be non-negative')
    if len(move_table)>1 and np.any(np.diff(move_table)<=0):
        raise ValueError('mv starts must be strictly increasing')
    events=[]; signal_size=len(signal)
    for i,start in enumerate(move_table):
        end=move_table[i+1] if i+1<len(move_table) else signal_size
        if start>=signal_size:
            events.append(np.empty(0,dtype=np.float32)); continue
        events.append(signal[int(start):min(int(end),signal_size)])
    return events

def _strand(is_reverse):
    return 'reverse' if is_reverse else 'forward'

def _gt_positions_in_read(chrom,strand,reference_start,reference_end):
    positions=_G_GT_POSITIONS.get(chrom,{}).get(strand)
    if positions is None or reference_start is None or reference_end is None:
        return np.empty(0,dtype=np.int64)
    left=np.searchsorted(positions,reference_start,side='left')
    right=np.searchsorted(positions,reference_end,side='left')
    return positions[left:right]

def find_cg_fragments(seq,move_table,signal,stride,aligned_pairs,align_data,ref_segment,ref_to_chr,methylation_data,signal_len=DEFAULT_SIGNAL_LEN,stats=None,gt_positions=None):
    """Extract read/reference/signal windows aligned to ground-truth CpGs.

    Each returned item is ``(read_seq, ref_seq, signal[31, L], stats[31, 5],
    position, label, strand_code)``. ``position`` uses 0-based genomic
    coordinates; ``strand_code`` is forward=0 and reverse=1.
    """
    fragments=[]; stats=stats if stats is not None else defaultdict(int)
    try:
        read_id,is_reverse,ref_name,initial_ref_start,_,_=align_data
        strand=_strand(is_reverse); strand_code=int(is_reverse)
        chrom=ref_to_chr.get(ref_name,{"chromosome":"unknown"})["chromosome"]
        query_events=build_query_events(signal,move_table,len(seq))
        valid_event=np.asarray([len(event)>0 for event in query_events],dtype=bool)
        signal_oob_count=int((~valid_event).sum())
        if signal_oob_count:
            stats['mv_signal_oob_bases']+=signal_oob_count; stats[f'mv_signal_oob_bases_{strand}']+=signal_oob_count
        ref_to_query={}
        for read_pos,ref_pos in aligned_pairs:
            if read_pos is not None and ref_pos is None:
                stats['insertion_bases']+=1; stats[f'insertion_bases_{strand}']+=1
            elif read_pos is None and ref_pos is not None:
                stats['deletion_bases']+=1; stats[f'deletion_bases_{strand}']+=1
            elif read_pos is not None and ref_pos is not None and 0<=read_pos<len(seq):
                ref_to_query[ref_pos]=read_pos
        if not ref_segment:
            stats['ref_fragment_missing']+=1; stats[f'ref_fragment_missing_{strand}']+=1
            return fragments
        if gt_positions is None:
            gt_positions=[]
            for idx_c in range(len(ref_segment)-1):
                if ref_segment[idx_c:idx_c+2]=='CG':
                    gt_positions.append(initial_ref_start+idx_c+(1 if is_reverse else 0))
        strand_labels=methylation_data.get(chrom,{}).get(strand,{})
        for label_pos in gt_positions:
            label_pos=int(label_pos)
            c_coord=label_pos-1 if is_reverse else label_pos
            local_c=c_coord-initial_ref_start
            if local_c<0 or local_c+2>len(ref_segment) or ref_segment[local_c:local_c+2]!='CG':
                stats['gt_reference_cpg_mismatch']+=1; stats[f'gt_reference_cpg_mismatch_{strand}']+=1; continue
            stats['candidate_bases']+=1; stats[f'candidate_bases_{strand}']+=1
            stats['candidate_cpg']+=1; stats[f'candidate_cpg_{strand}']+=1
            label=strand_labels.get(label_pos)
            if label is None:
                stats['label_missing']+=1; stats[f'label_missing_{strand}']+=1; continue
            center_query_pos=ref_to_query.get(label_pos)
            center_event_pos=(len(seq)-1-center_query_pos if is_reverse else center_query_pos) if center_query_pos is not None else None
            if center_query_pos is None or not valid_event[center_event_pos]:
                key='center_signal_oob_failed' if center_query_pos is not None else 'center_deletion_failed'
                stats[key]+=1; stats[f'{key}_{strand}']+=1; continue
            ref_start=label_pos-CONTEXT_BASES; ref_end=label_pos+CONTEXT_BASES+1
            local_start=ref_start-initial_ref_start; local_end=ref_end-initial_ref_start
            if local_start<0 or local_end>len(ref_segment):
                stats['context_bounds_failed']+=1; stats[f'context_bounds_failed_{strand}']+=1; continue
            ref_frag_forward=ref_segment[local_start:local_end]
            window_positions=list(range(ref_start,ref_end))
            window_oob_indices=[]
            for grid_idx,ref_pos in enumerate(window_positions):
                query_pos=ref_to_query.get(ref_pos)
                if query_pos is None:
                    continue
                event_pos=len(seq)-1-query_pos if is_reverse else query_pos
                if not valid_event[event_pos]:
                    output_idx=FRAGMENT_LENGTH-1-grid_idx if is_reverse else grid_idx
                    window_oob_indices.append(output_idx)
            if window_oob_indices:
                stats['window_signal_bounds_failed']+=1; stats[f'window_signal_bounds_failed_{strand}']+=1
                if max(window_oob_indices)>=CONTEXT_BASES:
                    stats['window_right_signal_bounds_failed']+=1; stats[f'window_right_signal_bounds_failed_{strand}']+=1
                continue
            read_bases=[]; frag_signal=[]; deletion_count=0
            for ref_pos in window_positions:
                query_pos=ref_to_query.get(ref_pos)
                if query_pos is None:
                    read_bases.append('N'); frag_signal.append(np.empty(0,dtype=np.float32)); deletion_count+=1
                else:
                    event_pos=len(seq)-1-query_pos if is_reverse else query_pos
                    read_bases.append(seq[query_pos]); frag_signal.append(query_events[event_pos])
            frag_seq=''.join(read_bases)
            ref_frag=ref_frag_forward
            if is_reverse:
                frag_seq=reverse_complement(frag_seq)
                ref_frag=reverse_complement(ref_frag)
                frag_signal=list(reversed(frag_signal))
            if len(frag_seq)!=FRAGMENT_LENGTH or len(ref_frag)!=FRAGMENT_LENGTH or len(frag_signal)!=FRAGMENT_LENGTH:
                stats['fragment_length_failed']+=1; stats[f'fragment_length_failed_{strand}']+=1; continue
            if ref_frag[CONTEXT_BASES]!='C':
                stats['ref_cpg_mismatch']+=1; stats[f'ref_cpg_mismatch_{strand}']+=1; continue
            if deletion_count:
                stats['fragments_with_deletion']+=1; stats[f'fragments_with_deletion_{strand}']+=1
                stats['deletion_bases_in_fragments']+=deletion_count; stats[f'deletion_bases_in_fragments_{strand}']+=deletion_count
            if calculate_sequence_similarity(frag_seq,ref_frag)<90.0:
                stats['low_similarity']+=1; stats[f'low_similarity_{strand}']+=1
            total_points=sum(len(s) for s in frag_signal); n_blocks=total_points//max(1,int(stride))
            if n_blocks<int(MIN_BLOCKS_LEN_FACTOR*FRAGMENT_LENGTH) or n_blocks>int(MAX_BLOCKS_LEN_FACTOR*FRAGMENT_LENGTH):
                stats['signal_length_filtered']+=1; stats[f'signal_length_filtered_{strand}']+=1
            fixed=pad_or_truncate_base_signals(frag_signal,signal_len); sig_stats=compute_base_signal_stats(frag_signal)
            empty=sum(1 for s in frag_signal if len(s)==0)
            stats['fragments_extracted']+=1; stats[f'fragments_extracted_{strand}']+=1
            stats['empty_base_signal_count']+=empty; stats[f'empty_base_signal_count_{strand}']+=empty
            fragments.append((frag_seq,ref_frag,fixed,sig_stats,label_pos,label,strand_code))
    except Exception as e:
        stats['find_cg_exceptions']+=1
        rid=align_data[0] if align_data and len(align_data)>0 else '?'
        logging.debug(f'find_cg_fragments for {rid}: {e}')
    return fragments

def _iter_bam_reads(bam, chromosome, stats):
    if chromosome is None:
        yield from bam.fetch(until_eof=True)
        return
    target_references=[ref for ref in bam.references if _G_REF_TO_CHR.get(ref,{}).get('chromosome')==chromosome]
    try:
        has_index=bam.has_index()
    except (AttributeError,ValueError):
        has_index=False
    if has_index:
        stats['bam_indexed_chromosome_fetch']+=1
        for reference in target_references:
            yield from bam.fetch(reference)
        return
    stats['bam_index_missing_fallback']+=1
    logging.warning(f'{os.path.basename(bam.filename.decode() if isinstance(bam.filename,bytes) else bam.filename)} has no index; falling back to a full BAM scan')
    yield from bam.fetch(until_eof=True)

def process_one_bam(task):
    bam_path=task['bam_path']; hdf5_path=task['hdf5_path']; signal_len=task.get('signal_len',DEFAULT_SIGNAL_LEN); chromosome=task.get('chromosome')
    results=[]; stats=defaultdict(int); h5=None; processed=set()
    scaling=_G_CONFIG['scaling']; offset=_G_CONFIG['offset']; stride=_G_CONFIG['constants']['STRIDE']
    try:
        h5=h5py.File(hdf5_path,'r') if hdf5_path else None
        h5_ids=h5 if chromosome is not None else (set(h5.keys()) if h5 else set())
        with pysam.AlignmentFile(bam_path,'rb') as bam:
            for read in _iter_bam_reads(bam,chromosome,stats):
                stats['reads_total']+=1
                chrom=_G_REF_TO_CHR.get(read.reference_name,{"chromosome":"unknown"})["chromosome"]
                if chromosome is not None and chrom!=chromosome: stats['reads_other_chromosome']+=1; continue
                if read.is_secondary or read.is_supplementary: stats['reads_secondary_or_supplementary']+=1; continue
                read_strand=_strand(read.is_reverse)
                gt_positions=_gt_positions_in_read(chrom,read_strand,read.reference_start,read.reference_end)
                if not len(gt_positions): stats['reads_without_gt_sites']+=1; stats[f'reads_without_gt_sites_{read_strand}']+=1; continue
                stats['reads_with_gt_sites']+=1; stats[f'reads_with_gt_sites_{read_strand}']+=1
                stats['gt_sites_in_read_intervals']+=len(gt_positions); stats[f'gt_sites_in_read_intervals_{read_strand}']+=len(gt_positions)
                if not read.has_tag('mv'): stats['reads_missing_mv']+=1; continue
                rid=read.query_name
                if rid not in h5_ids: stats['reads_missing_hdf5']+=1; continue
                try: raw=h5[f'/{rid}/Raw/Signal'][:]
                except Exception: stats['raw_signal_read_failed']+=1; continue
                if np.issubdtype(raw.dtype,np.integer): raw=convert_int_signal_to_physical(raw)
                signal=normalize_signal(raw,scaling,offset)
                mv=read.get_tag('mv')
                if isinstance(mv,(array.array,list,np.ndarray)): move=list(mv)
                elif isinstance(mv,str): move=mv.split(',')
                else: stats['mv_bad_type']+=1; continue
                seq=read.query_sequence
                if not seq: stats['reads_missing_sequence']+=1; continue
                seq=seq.upper()
                try:
                    mt=convert_move_table(np.asarray(move,dtype=np.int64),stride)
                    if len(mt)!=len(seq):
                        stats['mv_sequence_length_mismatch']+=1; continue
                    if len(mt)>1 and np.any(np.diff(mt)<=0):
                        stats['mv_non_increasing']+=1; continue
                    if len(mt) and mt[0]<0:
                        stats['mv_signal_negative_start']+=1; continue
                    oob_bases=int(np.count_nonzero(mt>=len(signal)))
                    if oob_bases:
                        stats['reads_with_mv_signal_oob']+=1
                        stats['mv_signal_oob_bases_detected']+=oob_bases
                except (TypeError,ValueError):
                    stats['mv_parse_failed']+=1; continue
                ref_seg=None
                if _W_REF_FASTA and read.reference_name and read.reference_start is not None:
                    try: ref_seg=_W_REF_FASTA.fetch(read.reference_name,read.reference_start,read.reference_end).upper()
                    except Exception: stats['ref_fetch_failed']+=1
                align_data=(rid,read.is_reverse,read.reference_name,read.reference_start,read.reference_end,len(seq))
                frags=find_cg_fragments(seq,mt,signal,stride,read.get_aligned_pairs(),align_data,ref_seg,_G_REF_TO_CHR,_G_METHYLATION_DATA,signal_len,stats,gt_positions)
                for frag_seq,ref_frag,frag_sig,sig_stats,pos,label,strand_code in frags:
                    key=(rid,pos)
                    if key in processed: stats['duplicate_read_pos']+=1; continue
                    processed.add(key); strand=_strand(bool(strand_code))
                    if _G_LABEL_MODE == 'continuous':
                        stats['continuous_labels']+=1; stats[f'continuous_labels_{strand}']+=1
                    else:
                        stats[f'label_{int(label)}']+=1; stats[f'label_{int(label)}_{strand}']+=1
                    stats[f'chr_{chrom}']+=1
                    stats['fragments_saved']+=1; stats[f'fragments_saved_{strand}']+=1
                    item=(encode_bases(frag_seq),encode_bases(ref_frag),frag_sig,sig_stats,label,pos,CHR_TO_CODE.get(chrom,0),strand_code)
                    results.append((chrom,item))
    except Exception as e:
        stats['process_one_bam_exceptions']+=1; logging.warning(f'process_one_bam({os.path.basename(bam_path)}): {e}')
    finally:
        if h5:
            try: h5.close()
            except Exception: pass
    return {'results':results,'stats':dict(stats),'bam_path':bam_path}

def _get_bam_base_name(filename):
    return filename[:-4] if filename.endswith('.bam') else filename

def _get_hdf5_base_name(filename):
    for ext in ('.hdf5','.h5'):
        if filename.endswith(ext): return filename[:-len(ext)]+'_pass'
    return filename

class DatasetGenerator:
    """Coordinate file matching, multiprocessing feature extraction, and chromosome-level PT shard output.

    ``label_mode='categorical'`` requires labels to be exactly 0 or 1;
    ``continuous`` accepts frequencies in ``[0, 1]`` and stores the labels
    tensor as float32.
    """

    def __init__(self,config_file,ref_mapping_file,methylation_pickle,hdf5_dir,bam_dir,output_dir,ref_fasta_path=None,signal_len=DEFAULT_SIGNAL_LEN,chromosome=None,label_mode='categorical'):
        output_path=Path(output_dir)
        if output_path.exists() and any(output_path.rglob('*.pt')):
            raise FileExistsError(f'Output directory already contains PT shards; refusing to mix old data: {output_dir}. Use a new directory.')
        os.makedirs(output_dir,exist_ok=True)
        logging.basicConfig(filename=os.path.join(output_dir,f"{os.path.basename(output_dir.rstrip('/'))}.log"),filemode='w',format='%(asctime)s - %(levelname)s - %(message)s',level=logging.INFO)
        with open(config_file,'r') as f: self.config=json.load(f)
        with open(ref_mapping_file,'r') as f: self.ref_to_chr=json.load(f)
        self.hdf5_dir=hdf5_dir; self.bam_dir=bam_dir; self.output_dir=output_dir; self.ref_fasta_path=ref_fasta_path; self.signal_len=signal_len; self.chromosome=chromosome
        self.methylation_data=self._load_pickle(methylation_pickle)
        if chromosome is not None:
            chromosome_data=self.methylation_data.get(chromosome)
            if chromosome_data is None:
                raise ValueError(f'Methylation pickle does not contain the target chromosome: {chromosome}')
            self.methylation_data={chromosome:chromosome_data}
        self.label_mode=label_mode
        self.label_dtype=torch.float32 if label_mode=='continuous' else torch.long
        self._validate_labels()
        global _G_METHYLATION_DATA,_G_REF_TO_CHR,_G_CONFIG,_G_LABEL_MODE,_G_GT_POSITIONS
        _G_METHYLATION_DATA=self.methylation_data; _G_REF_TO_CHR=self.ref_to_chr; _G_CONFIG=self.config; _G_LABEL_MODE=self.label_mode
        _G_GT_POSITIONS={
            chrom:{
                strand:np.asarray(sorted(strand_data.get(strand,{})),dtype=np.int64)
                for strand in ('forward','reverse')
            }
            for chrom,strand_data in self.methylation_data.items()
        }
    @staticmethod
    def _load_pickle(path):
        with open(path,'rb') as f:
            data=pickle.load(f)
        if not isinstance(data,dict):
            raise ValueError(f'Top-level methylation pickle object must be a dictionary: {path}')
        return data
    def _validate_labels(self):
        total=0; non_binary=0
        for chrom,strand_data in self.methylation_data.items():
            if not isinstance(strand_data,dict):
                raise ValueError(f'Chromosome data must be a dictionary: {chrom}')
            for strand_name in ('forward','reverse'):
                positions=strand_data.get(strand_name,{})
                if not isinstance(positions,dict):
                    raise ValueError(f'{chrom}/{strand_name} must be a position-to-label dictionary')
                for position,value in positions.items():
                    try: numeric=float(value); int(position)
                    except (TypeError,ValueError) as error:
                        raise ValueError(f'Invalid label: {chrom}/{strand_name}/{position}={value!r}') from error
                    if not np.isfinite(numeric) or not 0.0<=numeric<=1.0:
                        raise ValueError(f'Label is outside [0,1]: {chrom}/{strand_name}/{position}={numeric}')
                    total+=1; non_binary+=int(numeric not in (0.0,1.0))
        if not total: raise ValueError('Methylation pickle is empty')
        if self.label_mode=='categorical' and non_binary:
            raise ValueError(f'Categorical mode found {non_binary:,} continuous labels; correlation pickles require --label-mode continuous')
        logging.info(f'pickle labels: total={total:,}; non_binary={non_binary:,}; mode={self.label_mode}')
    def _build_tasks(self):
        bam_map={_get_bam_base_name(f):os.path.join(self.bam_dir,f) for f in os.listdir(self.bam_dir) if f.endswith('.bam') and not f.startswith('.')}
        h5_map={_get_hdf5_base_name(f):os.path.join(self.hdf5_dir,f) for f in os.listdir(self.hdf5_dir) if f.endswith(('.hdf5','.h5'))}
        tasks=[]; matched=0; unmatched=[]
        for base,bam_path in sorted(bam_map.items()):
            h5=h5_map.get(base); matched+=int(h5 is not None)
            if h5 is None: unmatched.append(base)
            tasks.append({'bam_path':bam_path,'hdf5_path':h5,'signal_len':self.signal_len,'chromosome':self.chromosome})
        logging.info(f'BAM files: {len(bam_map)}, HDF5 files: {len(h5_map)}, matched: {matched}, unmatched: {len(unmatched)}')
        if unmatched: logging.warning(f'Example unmatched BAM prefixes: {unmatched[:5]}')
        return tasks
    def run(self,batch_size=40000,num_workers=None):
        if num_workers is None: num_workers=min(32,cpu_count())
        tasks=[t for t in self._build_tasks() if t['hdf5_path'] is not None]
        if not tasks:
            print('Error: no usable (BAM, HDF5) file pairs'); return
        print(f'Starting {num_workers} workers for {len(tasks)} (BAM, HDF5) file pairs')
        start=time.time(); buffers=defaultdict(list); counters=defaultdict(int); total_stats=defaultdict(int); total=0; saved=0
        with Pool(num_workers,initializer=_worker_init,initargs=(self.ref_fasta_path,self.label_mode)) as pool:
            for out in tqdm(pool.imap_unordered(process_one_bam,tasks),total=len(tasks),desc='Processing file pairs'):
                for k,v in out.get('stats',{}).items(): total_stats[k]+=v
                for chrom,item in out.get('results',[]):
                    buffers[chrom].append(item); total+=1
                    if len(buffers[chrom])>=batch_size:
                        d=os.path.join(self.output_dir,chrom); os.makedirs(d,exist_ok=True)
                        self._save_batch(buffers[chrom],d,chrom,counters[chrom],self.label_dtype); saved+=len(buffers[chrom]); counters[chrom]+=1; buffers[chrom]=[]
        for chrom,rem in buffers.items():
            if rem:
                d=os.path.join(self.output_dir,chrom); os.makedirs(d,exist_ok=True)
                self._save_batch(rem,d,chrom,counters[chrom],self.label_dtype); saved+=len(rem); counters[chrom]+=1
        print(f'\nCompleted. Samples: {total:,}, batches: {sum(counters.values())}, elapsed: {time.time()-start:.1f}s')
        self._log_generation_stats(total_stats,total,saved)
    @staticmethod
    def _log_generation_stats(total_stats,total,saved):
        logging.info('='*60); logging.info('Dataset generation summary'); logging.info(f'fragments_buffered: {total:,}'); logging.info(f'fragments_written: {saved:,}')
        keys=['reads_total','reads_other_chromosome','bam_indexed_chromosome_fetch','bam_index_missing_fallback','reads_secondary_or_supplementary','reads_missing_mv','reads_missing_hdf5','raw_signal_read_failed','mv_bad_type','mv_parse_failed','mv_sequence_length_mismatch','mv_non_increasing','mv_signal_negative_start','reads_with_mv_signal_oob','mv_signal_oob_bases_detected','mv_signal_oob_bases','window_signal_bounds_failed','window_right_signal_bounds_failed','reads_missing_sequence','ref_fetch_failed','candidate_bases','candidate_cpg','insertion_bases','deletion_bases','center_deletion_failed','center_signal_oob_failed','fragments_with_deletion','deletion_bases_in_fragments','label_missing','ref_bounds_failed','ref_cpg_mismatch','context_bounds_failed','reverse_context_bounds_failed','fragment_bounds_failed','fragment_length_failed','ref_fragment_missing','low_similarity','signal_length_filtered','empty_base_signal_count','fragments_extracted','duplicate_read_pos','fragments_saved','find_cg_exceptions','process_one_bam_exceptions']
        for k in keys: logging.info(f'{k}: {int(total_stats.get(k,0)):,}')
        for strand in ('forward','reverse'):
            logging.info(f'--- strand={strand} ---')
            for k in keys:
                sk=f'{k}_{strand}'
                if sk in total_stats: logging.info(f'{sk}: {int(total_stats[sk]):,}')
        for prefix in ('label_','chr_'):
            for k in sorted(x for x in total_stats if x.startswith(prefix)): logging.info(f'{k}: {int(total_stats[k]):,}')
    @staticmethod
    def _save_batch(batch_data,chr_dir,chr_name,batch_idx,label_dtype=torch.long):
        read_kmer,ref_kmer,signals,stats,labels,pos,chr_codes,strands=zip(*batch_data)
        label_tensor=torch.tensor(labels,dtype=label_dtype)
        if not torch.isfinite(label_tensor).all() or not torch.all((label_tensor>=0)&(label_tensor<=1)):
            raise ValueError('PT labels must be finite values in [0,1]')
        if label_dtype==torch.long and not torch.all((label_tensor==0)|(label_tensor==1)):
            raise ValueError('Categorical PT labels must be 0 or 1')
        ds=TensorDataset(torch.tensor(read_kmer,dtype=torch.long),torch.tensor(ref_kmer,dtype=torch.long),torch.tensor(signals,dtype=torch.float),torch.tensor(stats,dtype=torch.float),label_tensor,torch.tensor(pos,dtype=torch.long),torch.tensor(chr_codes,dtype=torch.long),torch.tensor(strands,dtype=torch.long))
        torch.save(ds,os.path.join(chr_dir,f'{chr_name}_batch_{batch_idx}.pt'))

def main():
    import argparse
    p=argparse.ArgumentParser(description='Generate ModFormer training shards')
    p.add_argument('--methylation-pickle','--methylation_pickle',dest='methylation_pickle',required=True); p.add_argument('--hdf5-dir','--hdf5_dir',dest='hdf5_dir',required=True); p.add_argument('--bam-dir','--bam_dir',dest='bam_dir',required=True); p.add_argument('--output-dir','--output_dir',dest='output_dir',required=True)
    p.add_argument('--ref-fasta','--ref_fasta_path',dest='ref_fasta_path',required=True)
    p.add_argument('--config',required=True); p.add_argument('--ref-mapping','--ref_mapping',dest='ref_mapping',required=True)
    p.add_argument('--batch-size','--batch_size',dest='batch_size',type=int,default=40000); p.add_argument('--num-workers','--num_workers',dest='num_workers',type=int,default=8); p.add_argument('--signal-len','--signal_len',dest='signal_len',type=int,default=DEFAULT_SIGNAL_LEN)
    p.add_argument('--chromosome',choices=tuple(CHR_TO_CODE),help='Only output the specified chromosome, e.g. chr1 or chrX (default: all)')
    p.add_argument('--label-mode',choices=('categorical','continuous'),default='categorical',help='Use categorical for read/site classification pickles and continuous for correlation-frequency pickles')
    args=p.parse_args(); print('='*70); print('ModFormer dataset generation — read/reference k-mer + signal[31,16] + statistics[31,5]'); print('='*70)
    for k,v in vars(args).items(): print(f'  {k}: {v}')
    gen=DatasetGenerator(args.config,args.ref_mapping,args.methylation_pickle,args.hdf5_dir,args.bam_dir,args.output_dir,args.ref_fasta_path,args.signal_len,args.chromosome,args.label_mode)
    gen.run(args.batch_size,args.num_workers)
if __name__=='__main__': main()
