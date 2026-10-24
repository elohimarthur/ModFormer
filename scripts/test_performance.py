#!/usr/bin/env python3
"""
Performance profiling script for ModFormer inference pipeline.

Usage:
python scripts/test_performance.py sample.hdf5 sample.bam
python scripts/test_performance.py hdf5_directory bam_directory
"""

import sys
import os
import time
import logging
from pathlib import Path


# ============================================================
# Python path
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
modformer_dir = os.path.dirname(script_dir)

sys.path.insert(0, modformer_dir)
sys.path.insert(0, script_dir)


def find_hdf5_bam_pairs(hdf5_dir, bam_dir):
    """Match HDF5 files to BAM files using the inference naming convention."""
    pairs = []
    for hdf5_file in sorted(Path(hdf5_dir).iterdir()):
        if hdf5_file.suffix.lower() not in {'.hdf5', '.h5'}:
            continue
        bam_candidates = (
            Path(bam_dir) / f"{hdf5_file.stem}.bam",
            Path(bam_dir) / f"{hdf5_file.stem}_pass.bam",
        )
        bam_file = next((candidate for candidate in bam_candidates if candidate.exists()), None)
        if bam_file is not None:
            pairs.append((hdf5_file, bam_file))
    return pairs


def print_summary(logger, stats, title):
    """Log stage timings and throughput for one file or an aggregate run."""
    timing_keys = (
        'hdf5_read', 'bam_parse', 'reference_fetch',
        'fragment_extract', 'gpu_forward',
    )
    total = sum(stats[key] for key in timing_keys)
    logger.info('=' * 70)
    logger.info(title)
    logger.info('=' * 70)
    for key in timing_keys:
        elapsed = stats[key]
        fraction = elapsed / total * 100 if total else 0.0
        logger.info(f"{key:25s}{elapsed:8.2f}s {fraction:5.1f}%")
    logger.info('-' * 70)
    logger.info(f"Total {total:.2f}s")
    logger.info(f"Reads joined: {stats['reads_joined']}")
    logger.info(f"Fragments: {stats['candidate_fragments']}")
    logger.info(f"GPU batches: {stats['batches']}")
    if stats['fragment_extract']:
        logger.info(
            f"CPU production: "
            f"{stats['candidate_fragments'] / stats['fragment_extract']:.1f} fragments/s"
        )
    if stats['gpu_forward']:
        logger.info(
            f"GPU consumption: "
            f"{stats['candidate_fragments'] / stats['gpu_forward']:.1f} fragments/s"
        )
    logger.info('=' * 70)


def test_performance(hdf5_file, bam_file):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)


    # ========================================================
    # imports
    # ========================================================

    try:

        from predict import (
            HDF5Reader,
            parse_bam_alignment_results,
            find_cg_fragments,
            perform_batch_inference,
            load_model,
            SCALING,
            OFFSET,
            STRIDE
        )

        import torch
        import pysam
        import json


    except Exception as e:
        logger.error(f"Import failed: {e}")
        return



    # ========================================================
    # configuration
    # ========================================================

    reference_fasta = (
        "/home/data1/elohim/code/data/human_ref/hg38/"
        "GCF_000001405.40_GRCh38.p14_genomic.fna"
    )

    ref_mapping_file = (
        "/home/data1/elohim/code/modformer/"
        "output_mapping_v2.json"
    )

    model_path = (
        "/home/data1/elohim/code/modformer/"
        "ModFormer/checkpoints/modformer.ckpt"
    )


    stats = {

        "hdf5_read":0,
        "bam_parse":0,
        "reference_fetch":0,
        "fragment_extract":0,
        "batch_pack":0,
        "gpu_forward":0,

        "reads_in_hdf5":0,
        "valid_alignments":0,
        "reads_joined":0,

        "candidate_fragments":0,
        "batches":0,
    }



    try:

        # ====================================================
        # load resource
        # ====================================================

        logger.info("Loading resources")


        with open(ref_mapping_file) as f:
            ref_to_chr=json.load(f)


        import torch

        device=torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Device: {device}"
        )


        model=load_model(
            model_path,
            device
        )


        ref_handle=pysam.FastaFile(
            reference_fasta
        )


        # ====================================================
        # HDF5
        # ====================================================

        logger.info("[1/5] Reading HDF5")


        t=time.perf_counter()


        hdf5_reader=HDF5Reader(
            os.path.dirname(hdf5_file),
            SCALING,
            OFFSET,
            True
        )


        reads_data=hdf5_reader.read_hdf5_data(
            hdf5_file,
            logger
        )


        stats["hdf5_read"]=(
            time.perf_counter()-t
        )


        stats["reads_in_hdf5"]=len(reads_data)


        logger.info(
            f"Loaded {len(reads_data)} reads "
            f"in {stats['hdf5_read']:.2f}s"
        )



        # ====================================================
        # BAM
        # ====================================================

        logger.info("[2/5] Parsing BAM")


        t=time.perf_counter()


        alignment_results=parse_bam_alignment_results(
            bam_file,
            STRIDE,
            logger
        )


        stats["bam_parse"]=(
            time.perf_counter()-t
        )


        stats["valid_alignments"]=len(
            alignment_results
        )


        logger.info(
            f"Parsed {len(alignment_results)} alignments"
        )



        # ====================================================
        # Feature extraction
        # ====================================================

        logger.info(
            "[3/5] Extracting fragments"
        )


        batch_data=[]


        for read_id,alignment_info in alignment_results.items():


            if read_id not in reads_data:
                continue


            stats["reads_joined"]+=1


            read_data=reads_data[read_id]


            # ----------------------------
            # reference
            # ----------------------------

            t=time.perf_counter()


            try:

                ref_segment=ref_handle.fetch(
                    alignment_info["reference_name"],
                    alignment_info["reference_start"],
                    alignment_info["reference_end"]
                ).upper()


            except Exception:

                ref_segment=None


            stats["reference_fetch"] += (
                time.perf_counter()-t
            )



            # ----------------------------
            # fragment extraction
            # ----------------------------

            align_tuple=(

                read_id,
                alignment_info["is_reverse"],
                alignment_info["reference_name"],
                alignment_info["reference_start"],
                alignment_info["reference_end"],
                len(alignment_info["sequence"])

            )


            t=time.perf_counter()


            result=find_cg_fragments(

                alignment_info["query_sequence"],

                alignment_info["move_table"],

                read_data["signal"],

                alignment_info["aligned_pairs"],

                align_tuple,

                ref_to_chr,

                ref_segment,

                alignment_info["reference_start"]

            )


            stats["fragment_extract"] += (
                time.perf_counter()-t
            )



            if result is None:
                continue



            read_kmers, \
            ref_kmers, \
            signals, \
            sig_stats, \
            metadata = result



            n_frag=len(metadata)


            stats["candidate_fragments"] += n_frag



            # 保存batch

            batch_data.append(
                (
                    read_kmers,
                    ref_kmers,
                    signals,
                    sig_stats,
                    metadata
                )
            )



        logger.info(
            f"Fragments: {stats['candidate_fragments']}"
        )



        # ====================================================
        # GPU inference
        # ====================================================

        logger.info(
            "[4/5] GPU inference"
        )


        for item in batch_data:


            t=time.perf_counter()


            if device.type=="cuda":
                torch.cuda.synchronize()



            preds, metas, scores = perform_batch_inference(

                model,

                item,

                device,

                0.8

            )



            if device.type=="cuda":
                torch.cuda.synchronize()



            stats["gpu_forward"] += (
                time.perf_counter()-t
            )


            stats["batches"]+=1



        print_summary(logger, stats, f"PERFORMANCE REPORT: {os.path.basename(hdf5_file)}")
        return stats

    except Exception as e:

        logger.exception(
            f"Test failed: {e}"
        )
        return None

    finally:

        if "ref_handle" in locals():
            ref_handle.close()

        if "model" in locals():

            model.cpu()

            del model


        if "torch" in locals() and torch.cuda.is_available():

            torch.cuda.empty_cache()


def aggregate_stats(stats_list):
    """Combine numeric profiling counters across successfully tested files."""
    if not stats_list:
        return None
    aggregate = {key: 0 for key in stats_list[0]}
    for stats in stats_list:
        for key, value in stats.items():
            aggregate[key] += value
    return aggregate


def test_performance_directory(hdf5_dir, bam_dir):
    """Profile every matched HDF5/BAM pair in two directories."""
    logger = logging.getLogger(__name__)
    hdf5_dir = Path(hdf5_dir)
    bam_dir = Path(bam_dir)
    if not hdf5_dir.is_dir():
        raise NotADirectoryError(f"HDF5 directory does not exist: {hdf5_dir}")
    if not bam_dir.is_dir():
        raise NotADirectoryError(f"BAM directory does not exist: {bam_dir}")

    pairs = find_hdf5_bam_pairs(hdf5_dir, bam_dir)
    hdf5_count = sum(
        path.suffix.lower() in {'.hdf5', '.h5'}
        for path in hdf5_dir.iterdir()
    )
    logger.info(f"Found {hdf5_count} HDF5 files and {len(pairs)} matched BAM pairs")
    if hdf5_count != len(pairs):
        logger.warning(f"Skipping {hdf5_count - len(pairs)} HDF5 files without matching BAM")
    if not pairs:
        return None

    run_start = time.perf_counter()
    stats_list = []
    for index, (hdf5_file, bam_file) in enumerate(pairs, start=1):
        logger.info(f"[{index}/{len(pairs)}] Profiling {hdf5_file.name} <-> {bam_file.name}")
        stats = test_performance(str(hdf5_file), str(bam_file))
        if stats is not None:
            stats_list.append(stats)

    aggregate = aggregate_stats(stats_list)
    if aggregate is not None:
        aggregate['wall_clock'] = time.perf_counter() - run_start
        print_summary(
            logger, aggregate,
            f"AGGREGATE PERFORMANCE REPORT: {len(stats_list)}/{len(pairs)} files"
        )
        logger.info(f"Wall-clock directory time: {aggregate['wall_clock']:.2f}s")
    return aggregate


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python test_performance.py hdf5_file bam_file\n"
            "  python test_performance.py hdf5_directory bam_directory"
        )
        sys.exit(1)

    first_path, second_path = map(Path, sys.argv[1:])
    if first_path.is_dir() and second_path.is_dir():
        test_performance_directory(first_path, second_path)
    elif first_path.is_file() and second_path.is_file():
        test_performance(str(first_path), str(second_path))
    else:
        print("The two arguments must be either two files or two directories.")
        sys.exit(1)
