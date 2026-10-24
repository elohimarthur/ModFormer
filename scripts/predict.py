"""Run ModFormer read-level CpG modification inference on HDF5/BAM pairs.

Function: read Raw/Signal from HDF5 and sequence, alignment, and ``mv`` tags
from BAM; extract 31-nt CpG windows and run batched inference; support one-shot
execution and scheduled monitoring.
Inputs: HDF5 directory, BAM directory, reference FASTA, reference mapping, and
model checkpoint. Model inputs are read/reference [B,31], signals [B,31,16],
and stats [B,31,5].
Outputs: a read-level CSV and log for each file pair, plus SAM v1.6 MM/ML/MN tags.
Key constraints: only sites above the confidence threshold are emitted; ``mv``
block positions are multiplied by STRIDE to obtain signal indices. Tags are
written to a temporary BAM, which replaces the input BAM only after success.
"""

import argparse
import array
import asyncio
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
import pysam
import torch
from tqdm import tqdm

from modformer import ModFormer

# Use the spawn multiprocessing method to avoid CUDA multiprocessing issues
mp.set_start_method('spawn', force=True)

# Suppress HDF5 error output so corrupted files do not stall the process
h5py._errors.silence_errors()

# Global constants
SEQ_LEN = 31
SIGNAL_LEN = 16
SCAN_INTERVAL = 60  # 1 minute = 60 seconds
SAVE_INTERVAL = 100  # Save processed_files after every 100 files

# HDF5 signal-processing parameters, formerly read from config.json
SCALING = 1000
OFFSET = 0.0
STRIDE = 5

# Signal filtering constants
CLIP_VALUE = 5.0
CONTEXT_BASES = 15


def feistel_inverse(data: np.ndarray) -> np.ndarray:
    """按 int16 数据上的 Feistel 逆操作恢复原始整数信号。"""
    arr = np.asarray(data, dtype=np.int16)
    u = arr.astype(np.int32) + 32768
    L = (u >> 8) & 0xFF
    R = u & 0xFF
    g_L = (L * L) % 256
    L0 = R ^ g_L
    R0 = L
    v = ((L0 << 8) | R0).astype(np.int32)
    return (v - 32768).astype(np.int16)


def convert_int_signal_to_physical(signal_x):
    x = np.asarray(signal_x, dtype=np.float32)
    return (x * 2.5 / 6144.0) * (-200.0 / 11.0) / 4.0 / 2.5


def normalize_signal_v5(signal_raw, scaling, offset):
    """Normalize signal using MAD with clipping."""
    scaled = np.array(scaling * (signal_raw + offset), dtype=np.float32)
    median = np.median(scaled)
    mad = np.median(np.abs(scaled - median))
    if mad == 0:
        norm = scaled - median
    else:
        norm = (scaled - median) / mad
    return np.clip(norm, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)


def _reverse_complement(dna_sequence):
    """Return reverse-complement DNA sequence."""
    complement = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
        'N': 'N',
    }
    return ''.join(
        complement.get(base, 'N')
        for base in reversed(dna_sequence.upper())
    )


def _prepare_5mc_mm_context(seq, is_reverse):
    """
    Prepare the sequence context used for SAM MM/ML encoding.

    Important
    ---------
    pysam ``query_sequence`` is the BAM SEQ field.

    For FLAG 0x10 / read.is_reverse == True, BAM SEQ is already the
    reverse-complement of the sequence originally reported by the sequencer.

    MM coordinates, however, are defined relative to the original
    as-sequenced 5'->3' orientation.

    Therefore:
        forward alignment:
            as_sequenced_seq = BAM SEQ

        reverse alignment:
            as_sequenced_seq = reverse_complement(BAM SEQ)

    Returns
    -------
    (as_sequenced_seq, c_prefix_counts)
        c_prefix_counts[i] is the number of canonical C bases from the
        original read 5' end through position i, inclusive.
    """
    if seq is None:
        return None

    seq_upper = seq.upper()

    if is_reverse:
        as_sequenced_seq = _reverse_complement(seq_upper)
    else:
        as_sequenced_seq = seq_upper

    c_prefix_counts = np.cumsum(
        np.fromiter(
            (base == 'C' for base in as_sequenced_seq),
            dtype=np.int32,
            count=len(as_sequenced_seq),
        )
    )

    return as_sequenced_seq, c_prefix_counts


def _build_5mc_mm_tag_info(
    seq,
    bam_query_pos,
    is_reverse,
    mm_context=None,
):
    """
    Convert a BAM query coordinate into ONT/SAM-compatible 5mC MM coordinates.

    Parameters
    ----------
    seq
        BAM query_sequence / SAM SEQ.
    bam_query_pos
        Query coordinate returned by pysam alignment mapping.
        This coordinate is relative to BAM SEQ.
    is_reverse
        read.is_reverse / FLAG 0x10.
    mm_context
        Optional output of _prepare_5mc_mm_context().

    Returns
    -------
    ('C', '+', base_counter)

    ``base_counter`` is 1-based and counts canonical C bases from the
    original as-sequenced 5' end.

    Both forward and reverse alignments use C+m because the model calls
    modification on the strand that was actually sequenced. BAM FLAG does
    not determine the MM +/- field.
    """
    if seq is None:
        return None

    seq_upper = seq.upper()
    read_length = len(seq_upper)

    if bam_query_pos is None:
        return None

    bam_query_pos = int(bam_query_pos)

    if bam_query_pos < 0 or bam_query_pos >= read_length:
        return None

    if mm_context is None:
        mm_context = _prepare_5mc_mm_context(
            seq_upper,
            is_reverse,
        )

    if mm_context is None:
        return None

    as_sequenced_seq, c_prefix_counts = mm_context

    # BAM query coordinate -> original/as-sequenced coordinate
    if is_reverse:
        mm_query_pos = read_length - 1 - bam_query_pos
    else:
        mm_query_pos = bam_query_pos

    if (
        mm_query_pos < 0
        or mm_query_pos >= len(as_sequenced_seq)
    ):
        return None

    # A 5mC call must correspond to a canonical C in the original read.
    if as_sequenced_seq[mm_query_pos] != 'C':
        return None

    base_counter = int(
        c_prefix_counts[mm_query_pos]
    )

    if base_counter <= 0:
        return None

    # Always C+m for the directly sequenced strand.
    return (
        'C',
        '+',
        base_counter,
    )


def _probability_to_ml_byte(probability):
    """
    Convert modification probability [0,1] to SAM ML uint8 encoding.

    SAM ML uses 256 probability bins:
        N / 256 <= p < (N + 1) / 256

    p == 1 is clamped to 255.
    """
    probability = float(probability)

    if not np.isfinite(probability):
        raise ValueError(
            f'Non-finite modification probability: {probability}'
        )

    probability = min(
        1.0,
        max(0.0, probability),
    )

    return min(
        255,
        int(probability * 256.0),
    )


def _generate_mm_ml_5mc(
    positions_scores,
    as_sequenced_seq,
):
    """
    Generate ONT/SAM-compatible CpG 5mC MM/ML tags.

    Output MM group:
        C+m?,...

    '?' is intentional because this model evaluates CpG cytosines only.
    Canonical C bases skipped by the MM delta list may therefore be
    uncalled rather than confidently unmodified.

    positions_scores contains:
        (tag_base, tag_strand, base_counter, modification_probability)
    """
    if not as_sequenced_seq:
        return None, None

    as_sequenced_seq = as_sequenced_seq.upper()

    canonical_c_count = as_sequenced_seq.count('C')

    if canonical_c_count == 0:
        return None, None

    # key:
    #   zero-based index in the canonical-C sequence
    #
    # value:
    #   modification probability
    pos_to_score = {}

    for (
        tag_base,
        tag_strand,
        base_counter,
        score,
    ) in positions_scores:

        # Reject legacy C-m entries.
        if tag_base != 'C' or tag_strand != '+':
            continue

        canonical_index = int(base_counter) - 1

        if (
            canonical_index < 0
            or canonical_index >= canonical_c_count
        ):
            continue

        score = float(score)

        # Deduplicate a site if necessary.
        if (
            canonical_index not in pos_to_score
            or score > pos_to_score[canonical_index]
        ):
            pos_to_score[canonical_index] = score

    if not pos_to_score:
        return None, None

    sorted_positions = sorted(
        pos_to_score.keys()
    )

    deltas = []
    ml_scores = []

    previous_position = None

    for position in sorted_positions:

        if previous_position is None:
            # First value = number of canonical Cs skipped
            # from the original read 5' end.
            delta = position
        else:
            # Number of canonical Cs between two called sites.
            delta = (
                position
                - previous_position
                - 1
            )

        deltas.append(str(delta))

        ml_scores.append(
            _probability_to_ml_byte(
                pos_to_score[position]
            )
        )

        previous_position = position

    mm_value = (
        f"C+m?,{','.join(deltas)};"
    )

    ml_value = array.array(
        'B',
        ml_scores,
    )

    return mm_value, ml_value


def _build_query_by_ref(aligned_pairs, reference_start, reference_length):
    """Build a local reference-coordinate to query-coordinate NumPy map."""
    query_by_ref = np.full(reference_length, -1, dtype=np.int32)
    for read_pos, ref_pos in aligned_pairs:
        if read_pos is None or ref_pos is None:
            continue
        local_ref_pos = ref_pos - reference_start
        if 0 <= local_ref_pos < reference_length:
            query_by_ref[local_ref_pos] = read_pos
    return query_by_ref


def _build_event_bounds(move_table, signal_length, sequence_length):
    """Build signal event offsets without allocating one array per base."""
    if len(move_table) != sequence_length:
        raise ValueError(f'mv length {len(move_table)} != sequence length {sequence_length}')
    if sequence_length and move_table[0] < 0:
        raise ValueError('mv start must be non-negative')
    if len(move_table) > 1 and np.any(np.diff(move_table) <= 0):
        raise ValueError('mv starts must be strictly increasing')
    starts = np.asarray(move_table, dtype=np.int64)
    ends = np.empty_like(starts)
    if len(starts):
        ends[:-1] = starts[1:]
        ends[-1] = signal_length
    valid = (starts >= 0) & (starts < signal_length) & (ends > starts)
    return starts, np.minimum(ends, signal_length), valid


def _encode_sequence(sequence):
    """Encode one DNA sequence with the v6 five-symbol vocabulary."""
    table = np.full(256, 4, dtype=np.int8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    table[ord('N')] = 4
    return table[np.frombuffer(sequence.upper().encode('ascii'), dtype=np.uint8)]


def _extract_event_features(
    signal,
    event_positions,
    event_starts,
    event_ends,
    event_valid,
):
    """Create one v6 [31,16] signal tensor and [31,5] stats tensor."""
    fixed = np.zeros((SEQ_LEN, SIGNAL_LEN), dtype=np.float32)
    stats = np.zeros((SEQ_LEN, 5), dtype=np.float32)
    empty = np.empty(0, dtype=np.float32)
    for base_idx, event_pos in enumerate(event_positions):
        if event_pos < 0 or not event_valid[event_pos]:
            event = empty
        else:
            event = signal[event_starts[event_pos]:event_ends[event_pos]]
        n = event.size
        if n == 0:
            continue
        median = np.median(event)
        stats[base_idx] = (
            np.mean(event), np.std(event), median,
            np.median(np.abs(event - median)), np.log10(n + 1.0),
        )
        if n < SIGNAL_LEN:
            left = (SIGNAL_LEN - n) // 2
            fixed[base_idx, left:left + n] = event
        else:
            left = (n - SIGNAL_LEN) // 2
            fixed[base_idx] = event[left:left + SIGNAL_LEN]
    return fixed, stats


def find_cg_fragments(
    seq,
    move_table,
    signal,
    aligned_pairs,
    align_data,
    ref_to_chr,
    ref_sequence_segment,
    initial_read_ref_start,
):
    """Extract v6 fragments using array-based reference windows."""
    try:
        _, is_reverse, ref_name, _, _, _ = align_data
        chromosome = ref_to_chr.get(ref_name, {'chromosome': 'unknown'})['chromosome']
        if not ref_sequence_segment:
            return None

        mm_context = _prepare_5mc_mm_context(
            seq,
            is_reverse,
        )

        if mm_context is None:
            return None

        ref_length = len(ref_sequence_segment)
        query_by_ref = _build_query_by_ref(
            aligned_pairs, initial_read_ref_start, ref_length
        )
        event_starts, event_ends, event_valid = _build_event_bounds(
            move_table, len(signal), len(seq)
        )
        ref_codes = _encode_sequence(ref_sequence_segment)
        seq_codes = _encode_sequence(seq)
        ref_bytes = np.frombuffer(ref_sequence_segment.encode('ascii'), dtype=np.uint8)
        cpg_offsets = np.flatnonzero(
            (ref_bytes[:-1] == ord('C')) & (ref_bytes[1:] == ord('G'))
        )

        read_kmers = []
        ref_kmers = []
        signal_features = []
        stats_features = []
        metadata = []
        for cpg_offset in cpg_offsets:
            center_offset = int(cpg_offset + 1 if is_reverse else cpg_offset)
            window_start = center_offset - CONTEXT_BASES
            window_end = center_offset + CONTEXT_BASES + 1
            if window_start < 0 or window_end > ref_length:
                continue
            query_window = query_by_ref[window_start:window_end]
            center_query_pos = int(query_by_ref[center_offset])
            center_event_pos = (
                len(seq) - 1 - center_query_pos if is_reverse else center_query_pos
            )
            if center_query_pos < 0 or not event_valid[center_event_pos]:
                continue
            event_positions = query_window.copy()
            if is_reverse:
                event_positions = len(seq) - 1 - event_positions
            if is_reverse:
                event_positions = event_positions[::-1]
                missing_mask = (query_window < 0)[::-1]
            else:
                missing_mask = query_window < 0
            event_positions[missing_mask] = -1
            if np.any((event_positions >= 0) & ~event_valid[event_positions.clip(min=0)]):
                continue

            read_codes = seq_codes[query_window.clip(min=0)]
            read_codes[query_window < 0] = 4
            ref_window_codes = ref_codes[window_start:window_end]
            if is_reverse:
                read_codes = read_codes[::-1]
                ref_window_codes = ref_window_codes[::-1]
            fixed, stats = _extract_event_features(
                signal, event_positions, event_starts, event_ends, event_valid
            )
            tag_info = _build_5mc_mm_tag_info(
                seq=seq,
                bam_query_pos=center_query_pos,
                is_reverse=is_reverse,
                mm_context=mm_context,
            )

            if tag_info is None:
                continue
            # Store the actual original/as-sequenced coordinate in metadata.
            if is_reverse:
                original_center_pos = (
                    len(seq) - 1 - center_query_pos
                )
            else:
                original_center_pos = center_query_pos
            read_kmers.append(read_codes)
            ref_kmers.append(ref_window_codes)
            signal_features.append(fixed)
            stats_features.append(stats)

            metadata.append((
                chromosome,
                initial_read_ref_start + center_offset,
                align_data[0],
                is_reverse,
                original_center_pos,
                tag_info,
            ))
        if not metadata:
            return None
        return (
            np.asarray(read_kmers, dtype=np.int64),
            np.asarray(ref_kmers, dtype=np.int64),
            np.asarray(signal_features, dtype=np.float32),
            np.asarray(stats_features, dtype=np.float32),
            metadata,
        )
    except Exception:
        logging.exception(
            'find_cg_fragments failed for read %s',
            align_data[0] if align_data else 'unknown',
        )
        return None


def process_signal(
    signal,
    scaling,
    offset,
    convert_to_lab=True,
):
    """
    Convert and normalize HDF5 Raw/Signal.

    Processing
    ----------
    convert_to_lab=True:

        encrypted/commercial int16
            ↓
        feistel_inverse()
            ↓
        original integer signal
            ↓
        convert_int_signal_to_physical()
            ↓
        physical current
            ↓
        scaling / offset
            ↓
        MAD normalization

    convert_to_lab=False:

        integer Raw/Signal
            ↓
        convert_int_signal_to_physical()

    Floating-point Raw/Signal is assumed to already be in physical units.
    """
    try:
        signal_arr = np.asarray(
            signal
        )

        if convert_to_lab:

            # Feistel transform is defined on int16 values.
            signal_arr = feistel_inverse(
                signal_arr.astype(
                    np.int16,
                    copy=False,
                )
            )

            signal_arr = (
                convert_int_signal_to_physical(
                    signal_arr
                )
            )

        elif np.issubdtype(
            signal_arr.dtype,
            np.integer,
        ):
            signal_arr = (
                convert_int_signal_to_physical(
                    signal_arr
                )
            )

        else:
            signal_arr = np.asarray(
                signal_arr,
                dtype=np.float32,
            )

        normalized_signal = (
            normalize_signal_v5(
                signal_arr,
                scaling,
                offset,
            )
        )

        return (
            normalized_signal,
            None,
        )

    except Exception as e:
        return (
            None,
            str(e),
        )


def _test_hdf5_worker(hdf5_file, result_queue, max_problematic_reads):
    """Test HDF5 readability in an isolated subprocess."""
    problematic_read_ids = set()
    try:
        with h5py.File(hdf5_file, 'r') as f:
            all_read_ids = list(f.keys())
            if not all_read_ids:
                result_queue.put((False, set()))
                return

            for read_id in all_read_ids:
                if len(problematic_read_ids) >= max_problematic_reads:
                    result_queue.put((False, problematic_read_ids))
                    return

                try:
                    read_group = f[read_id]

                    if 'Raw' in read_group and 'Signal' in read_group['Raw']:
                        signal_dataset = read_group['Raw']['Signal']
                        _ = signal_dataset.shape
                        if signal_dataset.shape[0] > 0:
                            _ = signal_dataset[:]
                except Exception as e:
                    problematic_read_ids.add(read_id)

        result_queue.put((True, problematic_read_ids))
    except Exception:
        result_queue.put((False, problematic_read_ids))


def test_hdf5_read_all(
    hdf5_file,
    logger=None,
    max_problematic_reads=1,
    timeout_seconds=180,
):
    """Test an HDF5 file in an isolated subprocess with a hard timeout."""
    result_queue = mp.Queue()
    p = mp.Process(target=_test_hdf5_worker, args=(hdf5_file, result_queue, max_problematic_reads))
    p.start()
    p.join(timeout=timeout_seconds)

    if p.is_alive():
        # Terminate the subprocess when the timeout expires
        if logger:
            logger.error(
                f"HDF5 file test timed out after {timeout_seconds} seconds; "
                f"terminating and skipping: {os.path.basename(hdf5_file)}"
            )
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        return False, set()

    # Retrieve the subprocess result
    try:
        if not result_queue.empty():
            return result_queue.get_nowait()
        else:
            if logger:
                logger.error(f"HDF5 file test exited unexpectedly: {os.path.basename(hdf5_file)}")
            return False, set()
    except Exception:
        return False, set()

# ==================== GPU and model utilities ====================


def cleanup_gpu_memory(device):
    """Release cached GPU memory.

    Args:
        device: GPU device.
    """
    try:
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            gc.collect()
    except Exception as e:
        pass  # Ignore cleanup failures


def load_model(model_path, device):
    """Load a trained model.

    Args:
        model_path: Model checkpoint path.
        device: CPU or GPU device.

    Returns:
        Loaded model in evaluation mode.
    """
    seq_len = 31
    signal_len = 16
    num_classes = 2
    hidden_size = 1024
    vocab_size = 5
    embedding_size = 256

    model = ModFormer(
        seq_len=seq_len,
        signal_len=signal_len,
        stats_dim=5,
        num_classes=num_classes,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        elif k.startswith('model.'):
            new_k = k[len('model.'):]
        else:
            new_k = k
        if not new_k.startswith('criterion.'):
            new_state_dict[new_k] = v

    if not new_state_dict:
        raise ValueError("Model state dictionary is empty; check the model path and weight file.")

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible weights: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.device = device
    model.eval()
    return model


def safely_delete_model(model, device):
    """Delete a model and release its GPU memory.

    Args:
        model: Model to delete.
        device: Model device.
    """
    try:
        if model is not None:
            # Move the model to CPU
            model.cpu()
            # Delete the model reference
            del model
            # Release cached GPU memory
            cleanup_gpu_memory(device)
    except Exception as e:
        pass  # Ignore deletion failures


def apply_modification_tags(
    input_bam_file,
    modifications_list,
):
    """
    Add ONT/SAM-compatible 5mC MM/ML/MN tags to primary BAM records.

    Important
    ---------
    MM/ML coordinates are encoded relative to the original as-sequenced
    5'->3' read orientation, even when FLAG 0x10 is set and BAM SEQ is
    reverse-complemented.

    This CpG model emits:

        MM:Z:C+m?,...
        ML:B:C,...
        MN:i:<SEQ length>

    Both forward and reverse alignments use C+m.

    Parameters
    ----------
    input_bam_file
        BAM that will be replaced after successful writing/indexing.

    modifications_list
        List of:
            (
                read_id,
                tag_base,
                tag_strand,
                base_counter,
                modification_probability
            )
    """
    temp_bam_file = (
        f"{input_bam_file}."
        f"modformer_tmp.{os.getpid()}.bam"
    )

    temp_bai_file = temp_bam_file + '.bai'
    input_bai_file = input_bam_file + '.bai'

    # Group modification calls by read.
    mod_dict = defaultdict(list)

    for (
        read_id,
        tag_base,
        tag_strand,
        base_counter,
        score,
    ) in modifications_list:

        mod_dict[read_id].append((
            tag_base,
            tag_strand,
            base_counter,
            score,
        ))

    modified_reads_count = 0

    in_bam = None
    out_bam = None

    try:
        in_bam = pysam.AlignmentFile(
            input_bam_file,
            'rb',
        )

        out_bam = pysam.AlignmentFile(
            temp_bam_file,
            'wb',
            template=in_bam,
        )

        for read in in_bam.fetch(
            until_eof=True
        ):
            read_id = read.query_name

            # Do not synthesize tags for records without SEQ.
            if (
                read.query_sequence is None
                or len(read.query_sequence) == 0
            ):
                out_bam.write(read)
                continue

            # The inference pipeline operates on primary alignments.
            # Remove stale modification tags from secondary/supplementary
            # records to avoid carrying incompatible MM coordinates.
            if (
                read.is_secondary
                or read.is_supplementary
            ):
                for tag in (
                    'MM',
                    'ML',
                    'MN',
                ):
                    if read.has_tag(tag):
                        read.set_tag(
                            tag,
                            None,
                        )

                out_bam.write(read)
                continue

            if read_id not in mod_dict:

                # This inference run produced no modification calls for
                # this primary read.
                for tag in (
                    'MM',
                    'ML',
                    'MN',
                ):
                    if read.has_tag(tag):
                        read.set_tag(
                            tag,
                            None,
                        )

                out_bam.write(read)
                continue

            # Recover the original/as-sequenced read orientation.
            mm_context = _prepare_5mc_mm_context(
                read.query_sequence,
                read.is_reverse,
            )

            if mm_context is None:
                out_bam.write(read)
                continue

            as_sequenced_seq, _ = mm_context

            mm_value, ml_value = (
                _generate_mm_ml_5mc(
                    mod_dict[read_id],
                    as_sequenced_seq,
                )
            )

            if (
                mm_value is not None
                and ml_value is not None
                and len(ml_value) > 0
            ):
                read.set_tag(
                    'MM',
                    mm_value,
                    value_type='Z',
                )

                # array('B') causes pysam to write:
                #
                # ML:B:C,...
                read.set_tag(
                    'ML',
                    ml_value,
                )

                # MN stores SEQ length at the time MM/ML were produced.
                read.set_tag(
                    'MN',
                    len(read.query_sequence),
                    value_type='i',
                )

                modified_reads_count += 1

            else:
                # Avoid preserving stale modification tags.
                for tag in (
                    'MM',
                    'ML',
                    'MN',
                ):
                    if read.has_tag(tag):
                        read.set_tag(
                            tag,
                            None,
                        )

            out_bam.write(read)

    except Exception:
        # Close handles before deleting the temporary file.
        if out_bam is not None:
            out_bam.close()
            out_bam = None

        if in_bam is not None:
            in_bam.close()
            in_bam = None

        if os.path.exists(
            temp_bam_file
        ):
            os.remove(
                temp_bam_file
            )

        if os.path.exists(
            temp_bai_file
        ):
            os.remove(
                temp_bai_file
            )

        raise

    finally:
        if out_bam is not None:
            out_bam.close()

        if in_bam is not None:
            in_bam.close()

    # Validate that the temporary BAM can be indexed before replacing
    # the original.
    pysam.index(
        temp_bam_file
    )

    os.replace(
        temp_bam_file,
        input_bam_file,
    )

    if os.path.exists(
        temp_bai_file
    ):
        os.replace(
            temp_bai_file,
            input_bai_file,
        )

    else:
        if os.path.exists(
            input_bai_file
        ):
            os.remove(
                input_bai_file
            )

        pysam.index(
            input_bam_file
        )

    return modified_reads_count


def perform_batch_inference(model, batch_data, device, threshold):
    """Run inference directly from NumPy feature buffers and metadata."""
    try:
        read_kmers, ref_kmers, signals, signal_stats, metas = batch_data
        read_tensor = torch.as_tensor(read_kmers, dtype=torch.long, device=device)
        ref_tensor = torch.as_tensor(ref_kmers, dtype=torch.long, device=device)
        signal_tensor = torch.as_tensor(signals, dtype=torch.float32, device=device)
        stats_tensor = torch.as_tensor(signal_stats, dtype=torch.float32, device=device)

        with torch.inference_mode():
            outputs = model(read_tensor, ref_tensor, stats_tensor, signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            scores = torch.max(probabilities, dim=1).values.cpu().numpy()
            confident = scores > threshold

        return (
            predictions[confident],
            [meta for meta, keep in zip(metas, confident) if keep],
            scores[confident],
        )
    except Exception:
        logging.exception("ModFormer batch inference failed")
        return [], [], []


class HDF5Reader:
    """Read HDF5 ``Raw/Signal`` and apply unit conversion and MAD normalization.

    Sequence, alignment, and move-table data come from the corresponding BAM,
    not from HDF5.
    """

    def __init__(self, hdf5_dir, scaling, offset, convert_to_lab=True, validate_hdf5=False):
        """Initialize the HDF5 reader.

        Args:
            hdf5_dir: Directory containing HDF5 files.
            scaling: Signal scaling factor.
            offset: Signal offset.
            convert_to_lab: Convert signals to the laboratory representation.
            validate_hdf5: Fully test HDF5 readability before normal reading.
        """
        self.hdf5_dir = hdf5_dir
        self.scaling = scaling
        self.offset = offset
        self.convert_to_lab = convert_to_lab
        self.validate_hdf5 = validate_hdf5

    def read_hdf5_data(self, hdf5_file, logger=None):
        """Read all available signal records from one HDF5 file.

        Args:
            hdf5_file: HDF5 file path.
            logger: Optional logger.
        """
        reads_data = {}

        # Statistics
        total_reads = 0
        skipped_no_raw = 0
        skipped_signal_error = 0
        skipped_corrupted = 0  # Count corrupted records
        processed_reads = 0
        error_reads = 0

        try:
            if logger:
                logger.info(f"Reading HDF5 file: {os.path.basename(hdf5_file)}")

            problematic_reads = set()
            if self.validate_hdf5:
                is_readable, problematic_reads = test_hdf5_read_all(
                    hdf5_file, logger=logger
                )
                if not is_readable:
                    if logger:
                        logger.warning(
                            f"Full HDF5 test failed; continuing with per-read "
                            f"attempts: {os.path.basename(hdf5_file)}"
                        )
                elif problematic_reads:
                    if logger:
                        logger.error(
                            f"Full test found {len(problematic_reads)} problematic "
                            f"reads; skipping them and continuing: "
                            f"{os.path.basename(hdf5_file)}"
                        )
                        logger.error(
                            f"  Example problematic reads: "
                            f"{list(problematic_reads)[:5]}"
                        )

            with h5py.File(hdf5_file, 'r') as f:
                all_read_ids = list(f.keys())
                total_reads = len(all_read_ids)

                for read_id in all_read_ids:
                    try:
                        if read_id in problematic_reads:
                            skipped_corrupted += 1
                            continue

                        read_group = f[read_id]

                        # === Quickly check whether the required groups exist ===
                        try:
                            has_raw = 'Raw' in read_group and 'Signal' in read_group['Raw']
                        except Exception:
                            # Group-structure access failure indicates corrupted data
                            skipped_corrupted += 1
                            continue

                        if not has_raw:
                            skipped_no_raw += 1
                            continue

                        # === Read data; skip the record immediately if any step fails ===
                        try:
                            signal = read_group['Raw']['Signal'][:]
                        except Exception:
                            skipped_corrupted += 1
                            continue

                        # Process the signal
                        normalized_signal, signal_error = process_signal(
                            signal=signal,
                            scaling=self.scaling,
                            offset=self.offset,
                            convert_to_lab=self.convert_to_lab,
                        )

                        if normalized_signal is None:
                            skipped_signal_error += 1
                            continue

                        reads_data[read_id] = {
                            'signal': normalized_signal
                        }
                        processed_reads += 1

                    except (KeyError, OSError, RuntimeError) as e:
                        # Skip a corrupted or incomplete read and continue with the others
                        error_reads += 1
                        continue
                    except Exception as e:
                        # Catch all other exceptions so processing does not stall
                        error_reads += 1
                        continue

            # Report statistics
            if logger:
                logger.info(f"HDF5 file read complete: {os.path.basename(hdf5_file)}")
                logger.info(f"  - Total reads: {total_reads}")
                logger.info(f"  - Successfully processed: {processed_reads}")
                logger.info(f"  - Skipped (missing Raw/Signal): {skipped_no_raw}")
                logger.info(f"  - Skipped (signal-processing error): {skipped_signal_error}")
                logger.info(f"  - Skipped (corrupted data): {skipped_corrupted}")
                logger.info(f"  - Read errors: {error_reads}")

        except Exception as e:
            error_msg = f"Error reading HDF5 file: {e}"
            if logger:
                logger.error(error_msg)
            # Return records processed before the error rather than an empty dictionary

        return reads_data


def extract_move_table_from_mv_tag(read, stride):
    """Read per-base signal-start blocks from a BAM ``mv`` tag and multiply by stride.

    This inference script assumes that the ``mv`` tag does not encode stride;
    its contents are base-level block positions, such as ``mv:Z:0,12,23,...``
    or ``mv:B:i,0,12,23,...``. The final signal index is ``position * stride``.
    """
    if not read.has_tag('mv'):
        return None

    mv = read.get_tag('mv')
    if mv is None:
        return None

    try:
        if isinstance(mv, str):
            base_positions = [int(x) for x in mv.split(',') if x != '']
        else:
            base_positions = [int(x) for x in mv]
    except (TypeError, ValueError):
        return None

    if not base_positions:
        return None

    if read.query_sequence and len(base_positions) > len(read.query_sequence):
        base_positions = base_positions[:len(read.query_sequence)]

    return np.array(base_positions, dtype=np.int64) * int(stride)


def parse_bam_alignment_results(bam_file, stride, logger=None):
    """Parse alignments from a BAM file.

    Args:
        bam_file: BAM file path.
        stride: Conversion factor from ``mv`` block positions to raw-signal indices.
        logger: Optional logger.

    Returns:
        Alignment mapping ``{read_id: alignment_info}``.
    """
    alignment_results = {}
    total_reads = 0
    skipped_unmapped = 0
    skipped_secondary_or_supplementary = 0
    skipped_no_sequence = 0
    skipped_no_mv = 0
    skipped_bad_mv = 0
    parse_errors = 0

    try:
        with pysam.AlignmentFile(bam_file, 'rb') as bam:
            iterator = bam.fetch(until_eof=True)
            for read in iterator:
                total_reads += 1
                try:
                    if read.is_unmapped:
                        skipped_unmapped += 1
                        continue
                    if read.is_secondary or read.is_supplementary:
                        skipped_secondary_or_supplementary += 1
                        continue
                    if read.query_sequence is None:
                        skipped_no_sequence += 1
                        continue
                    if not read.has_tag('mv'):
                        skipped_no_mv += 1
                        continue

                    read_id = read.query_name
                    move_table = extract_move_table_from_mv_tag(read, stride)
                    if move_table is None or len(move_table) == 0:
                        skipped_bad_mv += 1
                        continue

                    alignment_results[read_id] = {
                        'is_reverse': read.is_reverse,
                        'reference_name': read.reference_name,
                        'reference_start': read.reference_start,
                        'reference_end': read.reference_end,
                        'query_sequence': read.query_sequence,
                        'sequence': read.query_sequence,
                        'move_table': move_table,
                        'aligned_pairs': read.get_aligned_pairs(),
                        'mapping_quality': read.mapping_quality
                    }
                except Exception:
                    parse_errors += 1
                    continue

    except Exception as e:
        if logger:
            logger.error(f"Failed to parse BAM file: {bam_file}: {e}")
            logger.error(traceback.format_exc())
        return alignment_results

    if logger:
        logger.info(f"BAM parsing statistics: {os.path.basename(bam_file)}")
        logger.info(f"  - Total BAM records: {total_reads}")
        logger.info(f"  - Valid alignments: {len(alignment_results)}")
        logger.info(f"  - Skipped (unmapped): {skipped_unmapped}")
        logger.info(f"  - Skipped (secondary/supplementary): {skipped_secondary_or_supplementary}")
        logger.info(f"  - Skipped (missing query_sequence): {skipped_no_sequence}")
        logger.info(f"  - Skipped (missing mv tag): {skipped_no_mv}")
        logger.info(f"  - Skipped (invalid mv tag): {skipped_bad_mv}")
        logger.info(f"  - Parse errors: {parse_errors}")

    return alignment_results


def feature_cpu_worker(
    file_queue,
    feature_queue,
    event_queue,
    ref_mapping_file,
    reference_fasta,
    batch_size,
    validate_hdf5,
    main_log_file,
):
    """Produce fixed-size model-input batches without loading a GPU model."""
    process_id = os.getpid()
    logger = _configure_worker_logger(f'feature_pid{process_id}', process_id, main_log_file)
    ref_fasta_handle = None

    try:
        with open(ref_mapping_file, 'r') as f:
            ref_to_chr = json.load(f)
        ref_fasta_handle = pysam.FastaFile(reference_fasta)
        logger.info("Feature worker started")

        while True:
            task = file_queue.get()
            if task is None:
                break

            file_id, hdf5_file, bam_file = task
            produced_batches = 0
            processed_reads = 0
            try:
                if not os.path.exists(bam_file):
                    raise FileNotFoundError(f"Corresponding BAM file does not exist: {bam_file}")

                reads_data = HDF5Reader(
                    os.path.dirname(hdf5_file), SCALING, OFFSET,
                    validate_hdf5=validate_hdf5
                ).read_hdf5_data(hdf5_file, logger)
                if not reads_data:
                    raise ValueError(f"HDF5 file contains no valid data: {hdf5_file}")

                alignment_results = parse_bam_alignment_results(bam_file, STRIDE, logger)
                batch_read_kmers = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
                batch_ref_kmers = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
                batch_signals = np.empty((batch_size, SEQ_LEN, SIGNAL_LEN), dtype=np.float32)
                batch_stats = np.empty((batch_size, SEQ_LEN, 5), dtype=np.float32)
                batch_metas = []
                batch_count = 0

                def submit_batch():
                    nonlocal produced_batches, batch_count, batch_metas
                    if batch_count == 0:
                        return
                    feature_queue.put((
                        'batch', file_id, batch_read_kmers[:batch_count].copy(),
                        batch_ref_kmers[:batch_count].copy(), batch_signals[:batch_count].copy(),
                        batch_stats[:batch_count].copy(), batch_metas,
                    ))
                    produced_batches += 1
                    batch_count = 0
                    batch_metas = []

                for read_id, alignment_info in alignment_results.items():
                    if read_id not in reads_data:
                        continue
                    processed_reads += 1
                    reference_start = alignment_info['reference_start']
                    try:
                        ref_sequence_segment = ref_fasta_handle.fetch(
                            alignment_info['reference_name'], reference_start,
                            alignment_info['reference_end']
                        ).upper()
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch reference sequence "
                            f"{alignment_info['reference_name']}:{reference_start}-"
                            f"{alignment_info['reference_end']}: {e}"
                        )
                        continue

                    fragments = find_cg_fragments(
                        alignment_info['query_sequence'], alignment_info['move_table'],
                        reads_data[read_id]['signal'], alignment_info['aligned_pairs'],
                        (
                            read_id, alignment_info['is_reverse'],
                            alignment_info['reference_name'], reference_start,
                            alignment_info['reference_end'], len(alignment_info['sequence']),
                        ),
                        ref_to_chr, ref_sequence_segment, reference_start
                    )
                    if fragments is None:
                        continue

                    read_kmers, ref_kmers, signals, stats, metas = fragments
                    offset = 0
                    while offset < len(metas):
                        take = min(batch_size - batch_count, len(metas) - offset)
                        end = offset + take
                        batch_read_kmers[batch_count:batch_count + take] = read_kmers[offset:end]
                        batch_ref_kmers[batch_count:batch_count + take] = ref_kmers[offset:end]
                        batch_signals[batch_count:batch_count + take] = signals[offset:end]
                        batch_stats[batch_count:batch_count + take] = stats[offset:end]
                        batch_metas.extend(metas[offset:end])
                        batch_count += take
                        offset = end
                        if batch_count == batch_size:
                            submit_batch()

                submit_batch()
                event_queue.put((
                    'file_complete', file_id, hdf5_file, bam_file,
                    processed_reads, produced_batches, None,
                ))
                logger.info(
                    f"Feature extraction completed for {os.path.basename(hdf5_file)}: "
                    f"{processed_reads} reads, {produced_batches} batches"
                )
            except Exception as e:
                logger.error(f"Feature extraction failed for {hdf5_file}: {e}")
                logger.error(traceback.format_exc())
                event_queue.put((
                    'file_complete', file_id, hdf5_file, bam_file,
                    processed_reads, produced_batches, str(e),
                ))
    except Exception as e:
        logger.error(f"Feature worker initialization failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        if ref_fasta_handle is not None:
            ref_fasta_handle.close()
        _close_worker_logger(logger)


def feature_gpu_worker(
    feature_queue,
    event_queue,
    model_path,
    threshold,
    gpu_id,
    main_log_file,
):
    """Consume feature batches, run GPU inference, and return predictions by file ID."""
    process_id = os.getpid()
    logger = _configure_worker_logger(
        f'inference_gpu{gpu_id}_pid{process_id}',
        process_id,
        main_log_file,
    )
    model = None
    device = None

    try:
        if torch.cuda.is_available() and gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
        logger.info(f"GPU inference worker started on {device}")
        model = load_model(model_path, device)
        logger.info(f"Model loaded on {device}")

        while True:
            item = feature_queue.get()
            if item is None:
                break
            _, file_id, read_kmers, ref_kmers, signals, stats, metas = item
            preds, confident_metas, scores = perform_batch_inference(
                model, (read_kmers, ref_kmers, signals, stats, metas), device, threshold
            )
            event_queue.put(('inference_complete', file_id, preds, confident_metas, scores))
    except Exception as e:
        logger.error(f"GPU inference worker failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        if model is not None:
            safely_delete_model(model, device)
        _close_worker_logger(logger)


def _configure_worker_logger(name, process_id, main_log_file):
    """Create an isolated file logger for one spawned worker."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(logging.NullHandler())
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(main_log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s - PID:{process_id} - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _close_worker_logger(logger):
    """Close handlers owned by a spawned worker logger."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def finalize_feature_queue_file(file_state, output_dir, main_logger):
    """Write one completed file's CSV and BAM tags after all batches are inferred."""
    hdf5_file = file_state['hdf5_file']
    bam_file = file_state['bam_file']
    file_identifier = os.path.splitext(os.path.basename(hdf5_file))[0]
    result_file = os.path.join(output_dir, f"{file_identifier}_result.csv")
    modifications_list = []

    with open(result_file, 'w') as f:
        f.write(
            "chromosome,position,read_id,is_reverse,pred,score,"
            "original_center_pos,cg_index\n"
        )
        for preds, metas, scores in file_state['inference_results']:
            for pred, score, meta in zip(preds, scores, metas):
                chromosome, position, read_id, is_reverse, center_pos, tag_info = meta
                f.write(
                    f"{chromosome},{position},{read_id},{is_reverse},{pred},"
                    f"{score:.4f},{center_pos},{tag_info[2]}\n"
                )
                tag_score = score if pred == 1 else 1.0 - score
                modifications_list.append(
                    (read_id, tag_info[0], tag_info[1], tag_info[2], tag_score)
                )

    if modifications_list:
        modified_count = apply_modification_tags(bam_file, modifications_list)
        main_logger.info(f"Added MM/ML tags to {modified_count} reads: {bam_file}")
    else:
        main_logger.info(f"No C sites found for {file_identifier}; BAM is unchanged")

    return {
        'file': file_identifier,
        'processed_reads': file_state['processed_reads'],
        'c_sites': len(modifications_list),
        'result_file': result_file,
        'bam_file': bam_file,
        'hdf5_file': hdf5_file,
    }


def process_single_hdf5_bam_pair(
    hdf5_file,
    bam_file,
    ref_to_chr,
    model,
    device,
    ref_fasta_handle,
    output_dir,
    batch_size,
    threshold,
    validate_hdf5=False,
    logger=None,
):
    """Process one HDF5/BAM pair, write CSV results, and add MM/ML tags to the BAM."""
    if logger is None:
        logger = logging.getLogger(__name__)

    hdf5_basename = os.path.splitext(os.path.basename(hdf5_file))[0]
    file_identifier = hdf5_basename
    result_file = os.path.join(output_dir, f"{hdf5_basename}_result.csv")

    if not os.path.exists(bam_file):
        logger.error(f"Corresponding BAM file does not exist: {bam_file}")
        return None

    hdf5_reader = HDF5Reader(
        os.path.dirname(hdf5_file), SCALING, OFFSET, validate_hdf5=validate_hdf5
    )
    modifications_list = []
    processed_reads = set()
    batch_read_kmers = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
    batch_ref_kmers = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
    batch_signals = np.empty((batch_size, SEQ_LEN, SIGNAL_LEN), dtype=np.float32)
    batch_stats = np.empty((batch_size, SEQ_LEN, 5), dtype=np.float32)
    batch_metas = []
    batch_count = 0
    processed_count = 0

    try:
        reads_data = hdf5_reader.read_hdf5_data(hdf5_file, logger)
        if not reads_data:
            logger.warning(f"HDF5 file contains no valid data: {hdf5_file}")
            return None
        logger.info(f"Read {len(reads_data)} records from {os.path.basename(hdf5_file)}")

        # Parse alignment results from the BAM file
        alignment_results = parse_bam_alignment_results(
            bam_file, STRIDE, logger
        )
        logger.info(f"Successfully parsed {len(alignment_results)} alignments")

        # Write the CSV result file
        with open(result_file, 'w') as f:
            # Write the header
            f.write(
                "chromosome,position,read_id,is_reverse,pred,score,"
                "original_center_pos,cg_index\n"
            )
            f.flush()

            def flush_batch():
                nonlocal batch_count, batch_metas
                if batch_count == 0:
                    return
                batch = (
                    batch_read_kmers[:batch_count], batch_ref_kmers[:batch_count],
                    batch_signals[:batch_count], batch_stats[:batch_count],
                    batch_metas,
                )
                preds, metas, scores = perform_batch_inference(
                    model, batch, device, threshold
                )
                for pred, score, meta in zip(preds, scores, metas):
                    (
                        chromosome_meta, position_meta, read_id_meta,
                        is_reverse_meta, original_center_pos, tag_info_meta,
                    ) = meta
                    f.write(
                        f"{chromosome_meta},{position_meta},{read_id_meta},"
                        f"{is_reverse_meta},{pred},{score:.4f},"
                        f"{original_center_pos},{tag_info_meta[2]}\n"
                    )
                    tag_score = score if pred == 1 else 1.0 - score
                    modifications_list.append((
                        read_id_meta, tag_info_meta[0], tag_info_meta[1],
                        tag_info_meta[2], tag_score,
                    ))
                f.flush()
                batch_count = 0
                batch_metas = []

            # Process each aligned read that is also present in HDF5
            for read_id in alignment_results:
                if read_id in processed_reads or read_id not in reads_data:
                    continue
                processed_reads.add(read_id)

                read_data = reads_data[read_id]
                alignment_info = alignment_results[read_id]

                # Build align_data_tuple
                align_data_tuple = (
                    read_id,
                    alignment_info['is_reverse'],
                    alignment_info['reference_name'],
                    alignment_info['reference_start'],
                    alignment_info['reference_end'],
                    len(alignment_info['sequence'])
                )

                # Fetch the reference segment using the shared ref_fasta_handle
                ref_sequence_segment = None
                current_initial_read_ref_start = alignment_info['reference_start']

                if (
                    ref_fasta_handle
                    and alignment_info['reference_name']
                    and current_initial_read_ref_start is not None
                    and alignment_info['reference_end'] is not None
                ):
                    try:
                        ref_sequence_segment = ref_fasta_handle.fetch(
                            alignment_info['reference_name'],
                            current_initial_read_ref_start,
                            alignment_info['reference_end'],
                        ).upper()
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch reference sequence "
                            f"{alignment_info['reference_name']}:"
                            f"{current_initial_read_ref_start}-"
                            f"{alignment_info['reference_end']}: {e}"
                        )

                # move_table is already converted from mv block positions to
                # signal indices using STRIDE.
                move_table_converted = alignment_info['move_table']

                # Find CpG fragments
                fragments = find_cg_fragments(
                    alignment_info['query_sequence'],
                    move_table_converted,
                    read_data['signal'],
                    alignment_info['aligned_pairs'],
                    align_data_tuple,
                    ref_to_chr,
                    ref_sequence_segment,
                    current_initial_read_ref_start
                )

                if fragments is not None:
                    read_kmers, ref_kmers, signals, stats, metas = fragments
                    offset = 0
                    while offset < len(metas):
                        take = min(batch_size - batch_count, len(metas) - offset)
                        end = offset + take
                        batch_read_kmers[batch_count:batch_count + take] = read_kmers[offset:end]
                        batch_ref_kmers[batch_count:batch_count + take] = ref_kmers[offset:end]
                        batch_signals[batch_count:batch_count + take] = signals[offset:end]
                        batch_stats[batch_count:batch_count + take] = stats[offset:end]
                        batch_metas.extend(metas[offset:end])
                        batch_count += take
                        offset = end
                        if batch_count == batch_size:
                            flush_batch()

                processed_count += 1

            # Process the remaining batch
            flush_batch()

        # If modifications were found, replace the original BAM with an
        # MM/ML-tagged version
        if modifications_list:
            logger.info(
                f"Adding MM/ML tags to the original BAM for {file_identifier} "
                f"({len(modifications_list)} C sites)..."
            )
            modified_count = apply_modification_tags(
                bam_file, modifications_list
            )
            logger.info(
                f"Added MM/ML tags to {modified_count} reads and updated the "
                f"original BAM: {bam_file}"
            )
        else:
            logger.info(
                f"No C sites found for {file_identifier}; the original BAM is "
                f"unchanged"
            )

        # Return the processing result
        logger.info(
            f"Completed {file_identifier}: {processed_count} reads, "
            f"{len(modifications_list)} C sites"
        )

        return {
            'file': file_identifier,
            'processed_reads': processed_count,
            'c_sites': len(modifications_list),
            # All C sites, including modified and unmodified sites
            'result_file': result_file,
            'bam_file': bam_file,
        }

    except Exception as e:
        logger.error(f"Failed to process HDF5/BAM pair {hdf5_file} + {bam_file}: {e}")
        logger.error(traceback.format_exc())
        return None


def chunk_list(lst, n):
    """Split a list into ``n`` approximately equal chunks.

    Args:
        lst: List to split.
        n: Number of chunks.

    Returns:
        List of chunks.
    """
    if n <= 0:
        return [lst]

    k, m = divmod(len(lst), n)
    chunks = []
    start = 0

    for i in range(n):
        chunk_size = k + (1 if i < m else 0)
        if chunk_size > 0:
            chunks.append(lst[start:start + chunk_size])
            start += chunk_size

    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def load_processed_files_record(record_file):
    """Load the set of processed files from a record file.

    Args:
        record_file: Record-file path.

    Returns:
        Set of processed files.
    """
    processed_files = set()
    if os.path.exists(record_file):
        try:
            with open(record_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_files.add(line)
        except Exception:
            pass  # Ignore read errors
    return processed_files


def append_processed_files_record(record_file, new_files):
    """Append newly processed files to the record.

    Args:
        record_file: Record-file path.
        new_files: List or set of newly processed files.
    """
    try:
        with open(record_file, 'a') as f:
            for file_path in new_files:
                f.write(f"{file_path}\n")
    except Exception:
        pass  # Ignore write errors


def scan_for_new_files(hdf5_dir, bam_dir, processed_files):
    """Scan for new HDF5/BAM file pairs."""
    hdf5_files = []
    for f in os.listdir(hdf5_dir):
        if f.endswith(('.hdf5', '.h5')):
            hdf5_files.append(os.path.join(hdf5_dir, f))

    new_hdf5_bam_pairs = []
    for hdf5_file in hdf5_files:
        if hdf5_file in processed_files:
            continue

        hdf5_basename = os.path.splitext(os.path.basename(hdf5_file))[0]
        bam_file = None
        possible_bam_names = [
            f"{hdf5_basename}.bam",
            f"{hdf5_basename}_pass.bam",
        ]
        for bam_name in possible_bam_names:
            candidate = os.path.join(bam_dir, bam_name)
            if os.path.exists(candidate):
                bam_file = candidate
                break

        if bam_file:
            new_hdf5_bam_pairs.append((hdf5_file, bam_file))

    return new_hdf5_bam_pairs


def check_stop_signal(output_dir):
    """Return whether ``stop.txt`` exists in the output directory."""
    return os.path.exists(os.path.join(output_dir, 'stop.txt'))


async def process_new_files_async(
    new_hdf5_bam_pairs, ref_mapping_file, reference_fasta, model_path, output_dir,
    batch_size, threshold, validate_hdf5, max_workers, gpu_ids, main_log_file, main_logger,
    processed_files_record=None, processed_files_set=None):
    """Run bounded CPU feature production and GPU inference as a streaming pipeline."""
    if not new_hdf5_bam_pairs:
        return []

    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    gpu_worker_count = max(1, min(max_workers or len(gpu_ids), len(gpu_ids)))
    cpu_worker_count = min(
        len(new_hdf5_bam_pairs), max(1, mp.cpu_count() - 1), max(2, gpu_worker_count * 8)
    )
    main_logger.info(
        f"Starting feature-queue pipeline: {cpu_worker_count} CPU feature workers, "
        f"{gpu_worker_count} GPU inference workers, {len(new_hdf5_bam_pairs)} file pairs"
    )

    ctx = mp.get_context('spawn')
    file_queue = ctx.Queue(maxsize=cpu_worker_count * 2)
    feature_queue = ctx.Queue(maxsize=gpu_worker_count * 4)
    event_queue = ctx.Queue()
    cpu_workers = [
        ctx.Process(
            target=feature_cpu_worker,
            args=(
                file_queue, feature_queue, event_queue, ref_mapping_file,
                reference_fasta, batch_size, validate_hdf5, main_log_file,
            ),
        )
        for _ in range(cpu_worker_count)
    ]
    gpu_workers = [
        ctx.Process(
            target=feature_gpu_worker,
            args=(
                feature_queue, event_queue, model_path, threshold,
                gpu_ids[i % len(gpu_ids)] if gpu_ids[0] is not None else None,
                main_log_file,
            ),
        )
        for i in range(gpu_worker_count)
    ]
    for worker in cpu_workers + gpu_workers:
        worker.start()

    loop = asyncio.get_running_loop()
    file_states = {
        file_id: {
            'hdf5_file': hdf5_file, 'bam_file': bam_file, 'processed_reads': 0,
            'expected_batches': None, 'received_batches': 0, 'error': None,
            'inference_results': [],
        }
        for file_id, (hdf5_file, bam_file) in enumerate(new_hdf5_bam_pairs)
    }

    async def submit_files():
        for file_id, (hdf5_file, bam_file) in enumerate(new_hdf5_bam_pairs):
            await loop.run_in_executor(None, file_queue.put, (file_id, hdf5_file, bam_file))
        for _ in cpu_workers:
            await loop.run_in_executor(None, file_queue.put, None)

    submission = asyncio.create_task(submit_files())
    finalized = set()
    all_results = []
    pending_processed_files = []
    processed_since_save = 0

    total_hdf5_files = len(file_states)
    with tqdm(
            total=total_hdf5_files,
            desc="HDF5 processed",
            ncols=100,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} HDF5 |{bar}| "
                "{percentage:3.0f}% [{elapsed}<{remaining}]"
            ),
    ) as pbar:
        while len(finalized) < len(file_states):
            try:
                event = await loop.run_in_executor(None, event_queue.get, True, 60)
            except queue.Empty:
                failed_workers = [
                    w.pid for w in cpu_workers + gpu_workers if not w.is_alive()
                ]
                if failed_workers:
                    raise RuntimeError(
                        f"Pipeline workers exited before completion: "
                        f"{failed_workers}"
                    )
                main_logger.warning("No pipeline event for 60 seconds; workers are still running")
                continue

            event_type, file_id, *payload = event
            state = file_states[file_id]
            if event_type == 'inference_complete':
                preds, metas, scores = payload
                state['inference_results'].append((preds, metas, scores))
                state['received_batches'] += 1
            elif event_type == 'file_complete':
                hdf5_file, bam_file, processed_reads, expected_batches, error = payload
                state.update(
                    hdf5_file=hdf5_file, bam_file=bam_file, processed_reads=processed_reads,
                    expected_batches=expected_batches, error=error,
                )
            else:
                main_logger.error(f"Unknown pipeline event: {event_type}")
                continue

            if (
                state['expected_batches'] is not None
                and state['received_batches'] == state['expected_batches']
                and file_id not in finalized
            ):
                finalized.add(file_id)
                if state['error']:
                    main_logger.error(f"Failed file: {state['hdf5_file']}: {state['error']}")
                else:
                    try:
                        result = finalize_feature_queue_file(state, output_dir, main_logger)
                        all_results.append(result)
                        pending_processed_files.append(result['hdf5_file'])
                        if processed_files_set is not None:
                            processed_files_set.add(result['hdf5_file'])
                        main_logger.info(
                            f"Completed file: {result['file']} - reads: "
                            f"{result['processed_reads']} - C sites: {result['c_sites']}"
                        )
                    except Exception as e:
                        main_logger.error(f"Failed to finalize {state['hdf5_file']}: {e}")
                        main_logger.error(traceback.format_exc())
                pbar.update(1)
                processed_since_save += 1
                if processed_since_save >= SAVE_INTERVAL and pending_processed_files:
                    append_processed_files_record(processed_files_record, pending_processed_files)
                    pending_processed_files = []
                    processed_since_save = 0

    await submission
    for worker in cpu_workers:
        worker.join()
    for _ in gpu_workers:
        feature_queue.put(None)
    for worker in gpu_workers:
        worker.join()

    if pending_processed_files and processed_files_record:
        append_processed_files_record(processed_files_record, pending_processed_files)
    main_logger.info(f"Feature-queue pipeline finished; collected {len(all_results)} results")
    return all_results


async def main_monitor_async(
    hdf5_dir,
    bam_dir,
    output_dir,
    ref_mapping_file,
    reference_fasta,
    model_path,
    batch_size=500,
    threshold=0.8,
    validate_hdf5=False,
    max_files=None,
    max_workers=None,
    gpu_ids=None,
    monitor_mode=True,
):
    """Periodically scan and process HDF5/BAM pairs, or run once and exit."""

    if not os.path.exists(hdf5_dir):
        raise FileNotFoundError(f"HDF5 directory does not exist: {hdf5_dir}")
    if not os.path.exists(bam_dir):
        raise FileNotFoundError(f"BAM directory does not exist: {bam_dir}")
    os.makedirs(output_dir, exist_ok=True)

    main_log_file = os.path.join(output_dir, 'main_inference.log')
    processed_files_record = os.path.join(output_dir, 'processed_files.txt')

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    file_handler = logging.FileHandler(main_log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    main_logger = logging.getLogger('main')
    main_logger.info("=" * 80)
    main_logger.info(
        "Starting methylation inference in monitoring mode"
        if monitor_mode
        else "Starting one-shot methylation inference"
    )
    main_logger.info(f"HDF5 directory: {hdf5_dir}")
    main_logger.info(f"BAM directory: {bam_dir}")
    main_logger.info(f"Output directory: {output_dir}")
    main_logger.info(f"Reference FASTA: {reference_fasta}")
    if monitor_mode:
        main_logger.info(f"Scan interval: {SCAN_INTERVAL} seconds")
    main_logger.info(f"Processed-file record: {processed_files_record}")
    main_logger.info(
        f"Full HDF5 preflight validation: "
        f"{'enabled' if validate_hdf5 else 'disabled'}"
    )
    main_logger.info(
        "HDF5 provides only Raw/Signal; sequence, alignment, and mv tags "
        "come directly from BAM"
    )
    main_logger.info("=" * 80)

    processed_files = load_processed_files_record(processed_files_record)
    main_logger.info(f"Loaded records for {len(processed_files)} processed files")

    scan_count = 0
    stop_signal_received = False

    try:
        while True:
            scan_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            main_logger.info(
                f"Starting scan {scan_count} [{current_time}]"
                if monitor_mode
                else f"Starting file scan and processing [{current_time}]"
            )

            if monitor_mode and check_stop_signal(output_dir):
                main_logger.info(
                    "Detected stop.txt; exiting after the current "
                    "processing cycle"
                )
                stop_signal_received = True

            new_hdf5_bam_pairs = scan_for_new_files(
                hdf5_dir, bam_dir, processed_files
            )
            if max_files and len(new_hdf5_bam_pairs) > max_files:
                main_logger.info(
                    f"Found {len(new_hdf5_bam_pairs)} new pairs; "
                    f"max_files={max_files}, processing only the first "
                    f"{max_files}"
                )
                new_hdf5_bam_pairs = new_hdf5_bam_pairs[:max_files]

            if new_hdf5_bam_pairs:
                main_logger.info(
                    f"Found {len(new_hdf5_bam_pairs)} new HDF5/BAM pairs:"
                )
                for hdf5_file, bam_file in new_hdf5_bam_pairs:
                    main_logger.info(
                        f"  - {os.path.basename(hdf5_file)} <-> "
                        f"{os.path.basename(bam_file)}"
                    )

                start_time = time.time()
                results = await process_new_files_async(
                    new_hdf5_bam_pairs,
                    ref_mapping_file,
                    reference_fasta,
                    model_path,
                    output_dir,
                    batch_size,
                    threshold,
                    validate_hdf5,
                    max_workers,
                    gpu_ids,
                    main_log_file,
                    main_logger,
                    processed_files_record,
                    processed_files,
                )
                end_time = time.time()

                successful_files = [r for r in results if r]
                total_reads = sum(r['processed_reads'] for r in successful_files)
                total_sites = sum(r['c_sites'] for r in successful_files)
                main_logger.info("Current scan complete:")
                main_logger.info(
                    f"  Successfully processed: "
                    f"{len(successful_files)}/{len(new_hdf5_bam_pairs)} file pairs"
                )
                main_logger.info(f"  Total reads: {total_reads}")
                main_logger.info(f"  Total C sites: {total_sites}")
                main_logger.info(
                    f"  Elapsed time: {end_time - start_time:.2f} seconds"
                )
                main_logger.info(
                    f"Processed-file record updated incrementally; "
                    f"{len(processed_files)} files recorded"
                )
            else:
                main_logger.info("No new file pairs found; skipping this cycle")

            if not monitor_mode:
                main_logger.info("One-shot mode complete; exiting")
                finish_file = os.path.join(output_dir, 'finish.txt')
                with open(finish_file, 'w') as f:
                    f.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("Execution mode: one-shot\n")
                    f.write(f"Total processed files: {len(processed_files)}\n")
                main_logger.info(f"Created completion marker: {finish_file}")
                break

            if stop_signal_received:
                main_logger.info("Current processing cycle complete; preparing to exit")
                finish_file = os.path.join(output_dir, 'finish.txt')
                with open(finish_file, 'w') as f:
                    f.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("Execution mode: monitoring\n")
                    f.write(f"Total scans: {scan_count}\n")
                    f.write(f"Total processed files: {len(processed_files)}\n")
                main_logger.info(f"Created completion marker: {finish_file}")
                break

            main_logger.info(
                f"Scan {scan_count} complete; waiting {SCAN_INTERVAL} seconds "
                f"before the next scan..."
            )
            main_logger.info("-" * 60)
            await asyncio.sleep(SCAN_INTERVAL)

        main_logger.info(
            "Monitoring mode exited normally"
            if monitor_mode
            else "One-shot mode exited normally"
        )

    except Exception as e:
        main_logger.error(f"Monitoring failed: {e}")
        main_logger.error(traceback.format_exc())

    finally:
        main_logger.info("=" * 80)
        main_logger.info("Monitoring ended")
        main_logger.info("=" * 80)


def main_monitor(
    hdf5_dir,
    bam_dir,
    output_dir,
    ref_mapping_file,
    reference_fasta,
    model_path,
    batch_size=500,
    threshold=0.8,
    validate_hdf5=False,
    max_files=None,
    max_workers=None,
    gpu_ids=None,
    monitor_mode=True,
):
    """Synchronous wrapper for the monitoring entry point."""
    return asyncio.run(main_monitor_async(
        hdf5_dir=hdf5_dir,
        bam_dir=bam_dir,
        output_dir=output_dir,
        ref_mapping_file=ref_mapping_file,
        reference_fasta=reference_fasta,
        model_path=model_path,
        batch_size=batch_size,
        threshold=threshold,
        validate_hdf5=validate_hdf5,
        max_files=max_files,
        max_workers=max_workers,
        gpu_ids=gpu_ids,
        monitor_mode=monitor_mode,
    ))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Methylation inference with one-shot and monitoring modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    One-shot mode (default):
        python inference_bam_v6.py --hdf5-dir /path/to/hdf5 --bam-dir /path/to/bam \
        --output-dir /path/to/output -M model.ckpt

    Monitoring mode:
        python inference_bam_v6.py --hdf5-dir /path/to/hdf5 --bam-dir /path/to/bam \
        --output-dir /path/to/output -M model.ckpt --monitor-mode

    Notes:
    HDF5 provides only Raw/Signal; sequence, alignment, and stride-free mv tags
    are read directly from BAM. mv values are converted to final signal indices
    using the built-in constant STRIDE=5.

    Stopping the program:
    In monitoring mode, create stop.txt in the output directory. The program
    stops after completing the current processing cycle.
            """
    )

    parser.add_argument(
        '--hdf5-dir', type=str, required=True,
        help='Directory containing HDF5 files',
    )
    parser.add_argument(
        '--bam-dir', type=str, required=True,
        help='Directory containing BAM files',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory',
    )
    parser.add_argument(
        '-r', '--ref-mapping-file', default="",
        help='Reference mapping file in JSON format',
    )
    parser.add_argument(
        '-f', '--reference-fasta', default="",
        help='Reference FASTA used by pysam',
    )
    parser.add_argument(
        '-M', '--model-path', default="",
        help='Trained model checkpoint',
    )
    parser.add_argument(
        '-b', '--batch-size', '--batch_size', dest='batch_size', type=int,
        default=2048, help='Inference batch size (default: 2048)',
    )
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5,
        help='Confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum files processed per scan (default: unlimited)',
    )
    parser.add_argument(
        '--max-workers', type=int, default=None,
        help='Maximum parallel workers (default: auto, up to 8)',
    )
    parser.add_argument(
        '--gpu-ids', type=int, nargs='*', default=None,
        help='GPU IDs, e.g. --gpu-ids 0 1 2 (default: auto-detect)',
    )
    parser.add_argument(
        '--cpu-only', action='store_true', help='Force CPU mode',
    )
    parser.add_argument(
        '--validate-hdf5', action='store_true',
        help='Fully test HDF5 readability before processing; reads '
             'Raw/Signal an extra time (default: disabled)',
    )
    parser.add_argument(
        '--monitor-mode', action='store_true', default=False,
        help=f'Enable continuous monitoring and scan every {SCAN_INTERVAL} '
             f'seconds (default: disabled; run once)',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    required_paths = [
        (args.ref_mapping_file, 'reference mapping file'),
        (args.reference_fasta, 'reference FASTA'),
        (args.model_path, 'model file'),
        (args.hdf5_dir, 'HDF5 directory'),
        (args.bam_dir, 'BAM directory'),
    ]
    for path, desc in required_paths:
        if not os.path.exists(path):
            print(f"Error: {desc} does not exist: {path}")
            sys.exit(1)

    gpu_ids = args.gpu_ids
    if args.cpu_only:
        gpu_ids = [None]
    elif gpu_ids is not None and len(gpu_ids) == 0:
        gpu_ids = None

    print(
        "Starting methylation inference in monitoring mode..."
        if args.monitor_mode
        else "Starting one-shot methylation inference..."
    )
    print(f"HDF5 directory: {args.hdf5_dir}")
    print(f"BAM directory: {args.bam_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reference mapping file: {args.ref_mapping_file}")
    print(f"Reference FASTA: {args.reference_fasta}")
    print(f"Model file: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Confidence threshold: {args.threshold}")
    print(
        "HDF5 signal mode: integer Raw/Signal values are always converted "
        "and MAD-normalized"
    )
    print(
        f"Full HDF5 preflight validation: "
        f"{'enabled' if args.validate_hdf5 else 'disabled'}"
    )
    print(f"Maximum files per scan: {args.max_files or 'unlimited'}")
    print(f"Maximum workers: {args.max_workers or 'auto-detect (up to 8)'}")
    print(f"GPU devices: {gpu_ids or 'auto-detect'}")
    if args.monitor_mode:
        print(f"Scan interval: {SCAN_INTERVAL} seconds")
        print(f"To stop the program, create: {args.output_dir}/stop.txt")
    print(
        "HDF5 read mode: Raw/Signal only; sequence, alignment, and mv tags "
        "come from BAM"
    )
    print(
        f"mv conversion: built-in STRIDE={STRIDE} converts block positions "
        f"to signal indices"
    )
    print("-" * 50)

    main_monitor(
        hdf5_dir=args.hdf5_dir,
        bam_dir=args.bam_dir,
        output_dir=args.output_dir,
        ref_mapping_file=args.ref_mapping_file,
        reference_fasta=args.reference_fasta,
        model_path=args.model_path,
        batch_size=args.batch_size,
        threshold=args.threshold,
        validate_hdf5=args.validate_hdf5,
        max_files=args.max_files,
        max_workers=args.max_workers,
        gpu_ids=gpu_ids,
        monitor_mode=args.monitor_mode,
    )
