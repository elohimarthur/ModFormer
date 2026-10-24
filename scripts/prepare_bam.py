#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare per-file BAMs and move-table tags for ModFormer data generation.

Function: align each FASTQ independently to the reference genome, read base
start positions from the matching ``.path`` file, and write them to the BAM
``mv`` tag.
Inputs: FASTQ directory, path directory, reference FASTA, and minimap2 index.
Outputs: ``output_dir/bam/<base_name>.bam``, its BAI index, a processing log,
and ``manifest.json`` mapping base names to BAM paths.
Key constraints: output BAMs are coordinate-sorted and minimap2 secondary
alignments are disabled; ``mv`` stores block positions without stride, so later
steps must use the same stride for conversion.
"""

import os
import subprocess
import json
import argparse
import logging
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _path_iterator(path_file):
    """Read ``(read_id, base_positions_str)`` records from a ``.path`` file."""
    with open(path_file, 'r') as f:
        read_id = None
        base_positions = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if read_id is not None:
                    yield read_id, ','.join(map(str, base_positions))
                read_id = line[1:]
                base_positions = []
            else:
                move_table = ''.join(c for c in line if c in '01')
                base_positions.extend(i for i, s in enumerate(move_table) if s == '1')
        if read_id is not None:
            yield read_id, ','.join(map(str, base_positions))


def _get_base_name(filename, file_type):
    """
    Extract the file base name.
    Append ``_pass`` to HDF5 base names to match FASTQ/path naming conventions.
    """
    extensions = ['.fastq.gz', '.fq.gz', '.hdf5', '.h5', '.fastq', '.fq', '.path']
    for ext in extensions:
        if filename.endswith(ext):
            base = filename[:-len(ext)]
            break
    else:
        base = filename
    if file_type == 'hdf5':
        return base + '_pass'
    return base


def process_single_fastq(task, reference_mmi, minimap2_threads):
    """
    Process one FASTQ: align with minimap2 and inject ``mv`` tags.
    Return ``(base_name, bam_path, n_with_mv, success, diag_info)``.
    ``diag_info`` contains detailed diagnostics for each processing stage.
    """
    import pysam

    base_name = task['base_name']
    fastq_path = task['fastq_path']
    path_file = task['path_path']
    output_bam = task['output_bam']

    bam_dir = os.path.dirname(output_bam)
    temp_sorted_bam = os.path.join(bam_dir, f'.tmp_{base_name}_sorted.bam')

    n_with_mv = 0
    diag = {
        'base_name': base_name,
        'path_file': path_file,
        'path_exists': path_file is not None and os.path.exists(path_file) if path_file else False,
        'path_entries': 0,
        'bam_total_reads': 0,
        'bam_matched_reads': 0,
        'bam_sample_ids': [],
        'path_sample_ids': [],
        'error': None,
    }

    try:
        # --- 1. minimap2 alignment and samtools sorting ---
        cmd = (
            f"minimap2 -ax map-ont {reference_mmi} {fastq_path} "
            f"-y -t {minimap2_threads} --secondary=no 2>/dev/null "
            f"| samtools sort -o {temp_sorted_bam}"
        )
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            diag['error'] = f"minimap2/samtools returned nonzero status: {ret}"
            return base_name, output_bam, 0, False, diag

        # --- 2. Load the read-to-mv mapping from the path file ---
        read_id_to_mv = {}
        if path_file and os.path.exists(path_file):
            for rid, mv_str in _path_iterator(path_file):
                read_id_to_mv[rid] = mv_str
        diag['path_entries'] = len(read_id_to_mv)
        diag['path_sample_ids'] = list(read_id_to_mv.keys())[:3]

        if not read_id_to_mv:
            diag['error'] = (
                f"path mapping is empty! path_file={path_file}, "
                f"path_exists={diag['path_exists']}"
            )

        # --- 3. Inject mv tags and write the final BAM ---
        with pysam.AlignmentFile(temp_sorted_bam, 'rb') as bam_in, \
             pysam.AlignmentFile(output_bam, 'wb', header=bam_in.header) as bam_out:
            for read in bam_in:
                diag['bam_total_reads'] += 1
                if diag['bam_total_reads'] <= 3:
                    diag['bam_sample_ids'].append(read.query_name)
                mv_value = read_id_to_mv.get(read.query_name)
                if mv_value:
                    read.set_tag('mv', mv_value)
                    n_with_mv += 1
                bam_out.write(read)

        diag['bam_matched_reads'] = n_with_mv

        # --- 4. Build the BAM index ---
        pysam.index(output_bam)

        return base_name, output_bam, n_with_mv, True, diag

    except Exception as e:
        import traceback
        diag['error'] = traceback.format_exc()
        return base_name, output_bam, n_with_mv, False, diag

    finally:
        if os.path.exists(temp_sorted_bam):
            os.remove(temp_sorted_bam)


class BatchBAMBuilder:
    """Build ``mv``-tagged BAMs for matched FASTQ/path pairs in parallel."""

    def __init__(self, pass_dir, path_dir, output_dir, reference_mmi, ref_file,
                 n_workers=None, minimap2_threads=2):
        self.pass_dir = pass_dir
        self.path_dir = path_dir
        self.output_dir = output_dir
        self.reference_mmi = reference_mmi
        self.ref_file = ref_file
        self.minimap2_threads = minimap2_threads

        if n_workers is None:
            total_cpu = os.cpu_count() or 4
            self.n_workers = max(1, total_cpu // (minimap2_threads + 1))
        else:
            self.n_workers = n_workers

        self.bam_dir = os.path.join(output_dir, 'bam')
        os.makedirs(self.bam_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'mv2bam_v3.log'), mode='w'),
                logging.StreamHandler(),
            ]
        )

    def _ensure_mmi_index(self):
        if not os.path.exists(self.reference_mmi):
            logging.info(f"Building minimap2 index: {self.reference_mmi}")
            subprocess.check_call(
                f"minimap2 -d {self.reference_mmi} {self.ref_file}",
                shell=True
            )

    def _build_task_list(self):
        fastq_files = [f for f in os.listdir(self.pass_dir)
                       if f.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz'))]
        path_files_set = set(os.listdir(self.path_dir)) if os.path.isdir(self.path_dir) else set()

        logging.info(f"FASTQ directory: {self.pass_dir} ({len(fastq_files)} files)")
        logging.info(f"Path directory: {self.path_dir} ({sum(1 for f in path_files_set if f.endswith('.path'))} .path files)")

        fastq_map = {}
        for f in fastq_files:
            base = _get_base_name(f, 'fastq')
            fastq_map[base] = os.path.join(self.pass_dir, f)

        path_map = {}
        for f in path_files_set:
            if f.endswith('.path'):
                base = _get_base_name(f, 'path')
                path_map[base] = os.path.join(self.path_dir, f)

        # Log a few base-name examples to diagnose matching problems
        fq_samples = list(fastq_map.keys())[:3]
        path_samples = list(path_map.keys())[:3]
        logging.info(f"FASTQ base_name examples: {fq_samples}")
        logging.info(f"Path base_name examples: {path_samples}")

        matched = 0
        no_path = 0
        skipped_existing = 0
        tasks = []
        for base_name, fq_path in fastq_map.items():
            bam_path = os.path.join(self.bam_dir, f'{base_name}.bam')
            if os.path.exists(bam_path) and os.path.exists(bam_path + '.bai'):
                skipped_existing += 1
                continue
            path_path = path_map.get(base_name)
            if path_path:
                matched += 1
            else:
                no_path += 1
            tasks.append({
                'base_name': base_name,
                'fastq_path': fq_path,
                'path_path': path_path,
                'output_bam': bam_path,
            })

        logging.info(f"Task summary: new={len(tasks)}, skipped_existing={skipped_existing}, "
                     f"path_matched={matched}, path_unmatched={no_path}")
        if no_path > 0:
            no_path_examples = [bn for bn in fastq_map if path_map.get(bn) is None][:5]
            logging.warning(f"Example base names without path matches: {no_path_examples}")

        return tasks

    def run(self):
        start_time = time.time()

        self._ensure_mmi_index()
        tasks = self._build_task_list()
        logging.info(f"Pending tasks: {len(tasks)}, parallel workers: {self.n_workers}")

        if not tasks:
            logging.info("All BAM files already exist; skipping processing.")
            self._write_manifest()
            return

        worker_fn = partial(
            process_single_fastq,
            reference_mmi=self.reference_mmi,
            minimap2_threads=self.minimap2_threads,
        )

        success_count = 0
        fail_count = 0
        total_mv = 0
        zero_mv_count = 0
        empty_path_count = 0
        logged_detail = 0

        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.imap_unordered(worker_fn, tasks)
            for base_name, bam_path, n_mv, ok, diag in tqdm(results, total=len(tasks),
                                                              desc="Building BAMs"):
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                total_mv += n_mv

                if n_mv == 0:
                    zero_mv_count += 1
                if diag.get('path_entries', 0) == 0:
                    empty_path_count += 1

                # Log detailed diagnostics for the first five or problematic tasks
                if logged_detail < 5 or not ok or n_mv == 0:
                    logging.info(
                        f"[{base_name}] ok={ok}, "
                        f"bam_reads={diag['bam_total_reads']}, "
                        f"path_entries={diag['path_entries']}, "
                        f"mv_injected={n_mv}, "
                        f"path_file={diag['path_file']}, "
                        f"path_exists={diag['path_exists']}"
                    )
                    if diag['bam_sample_ids'] or diag['path_sample_ids']:
                        logging.info(
                            f"  Example BAM read IDs: {diag['bam_sample_ids']}"
                        )
                        logging.info(
                            f"  Example path read IDs: {diag['path_sample_ids']}"
                        )
                    if diag.get('error'):
                        logging.error(f"  Error: {diag['error']}")
                    logged_detail += 1

        self._write_manifest()

        elapsed = time.time() - start_time
        logging.info("=" * 60)
        logging.info(f"Completed: {success_count}/{len(tasks)} succeeded, {fail_count} failed")
        logging.info(f"Total mv tags injected: {total_mv:,}")
        logging.info(f"Files with mv=0: {zero_mv_count}/{len(tasks)}")
        logging.info(f"Files with empty path mappings: {empty_path_count}/{len(tasks)}")
        logging.info(f"Total elapsed time: {elapsed:.1f}s")
        logging.info("=" * 60)

    def _write_manifest(self):
        bam_files = [f for f in os.listdir(self.bam_dir) if f.endswith('.bam') and not f.startswith('.')]
        manifest = {}
        for f in sorted(bam_files):
            base = _get_base_name(f, 'fastq')
            if base.endswith('.bam'):
                base = base[:-4]
            manifest[base] = os.path.join(self.bam_dir, f)

        manifest_path = os.path.join(self.output_dir, 'manifest.json')
        with open(manifest_path, 'w') as fh:
            json.dump(manifest, fh, indent=2)
        logging.info(f"Manifest written: {manifest_path} ({len(manifest)} entries)")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build per-sample BAM files and inject mv tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--pass_dir', '-p', required=True, help='Directory containing FASTQ files')
    parser.add_argument('--path_dir', '-a', required=True, help='Directory containing path files')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    parser.add_argument('--ref_file', '-r',
                        default='/home/data1/elohim/code/data/human_ref/hg38/GCF_000001405.40_GRCh38.p14_genomic.fna',
                        help='Reference genome FASTA')
    parser.add_argument('--reference_mmi', '-m',
                        default='/home/data1/elohim/code/data/human_ref/hg38/human.mmi',
                        help='minimap2 index file')
    parser.add_argument('--n_workers', '-n', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--minimap2_threads', '-t', type=int, default=2,
                        help='Threads per minimap2 process (default: 2)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("=" * 60)
    print("mv2bam v3 - per-file parallel alignment and mv-tag injection")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    builder = BatchBAMBuilder(
        pass_dir=args.pass_dir,
        path_dir=args.path_dir,
        output_dir=args.output_dir,
        reference_mmi=args.reference_mmi,
        ref_file=args.ref_file,
        n_workers=args.n_workers,
        minimap2_threads=args.minimap2_threads,
    )
    builder.run()
