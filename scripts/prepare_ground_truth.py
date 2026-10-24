#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge EpiQC technical replicates and generate ModFormer ground truth.

Function: merge two technical replicates for each of six sequencing methods,
filter sites within each method by coverage and methylation frequency, and then
intersect the sites across all six methods.
Inputs: EpiQC bedGraph files using 0-based, half-open BED coordinates.
Outputs: read-level categorical labels, site-level categorical labels,
continuous frequencies for correlation, and corresponding BED, TSV, pickle,
intermediate merged files, and statistical reports.
Key constraints: by default, retain only autosomes and require coverage in both
replicates; forward-strand labels use the CpG interval start and reverse-strand
labels use end-1. Continuous frequency is the arithmetic mean across six methods.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

METHODS = ("TruSeq", "TrueMethylBS", "TrueMethylOX", "EMSeq", "MethylSeq", "SPLAT")
GROUND_TRUTHS = {
    "train_read": {
        "min_coverage": 10,
        "positive_threshold": 0.9,
        "negative_threshold": 0.1,
        "negative_inclusive": False,
    },
    "site_level": {
        "min_coverage": 10,
        "positive_threshold": 0.8,
        "negative_threshold": 0.2,
        "negative_inclusive": False,
    },
}
AUTOSOME = re.compile(r"^chr(?:[1-9]|1[0-9]|2[0-2])$")
Site = tuple[str, int, str]
Counts = tuple[float, int, int, int]


class StrictMethylationFilter:
    """Validate EpiQC inputs, merge replicates, intersect six methods, and export labels."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.data_dir = Path(args.data_dir).expanduser().resolve()
        self.genome = args.genome
        self.progress_every = args.progress_every
        self.require_both_replicates = not args.allow_single_replicate_sites
        self.output_prefix = Path(args.output_prefix).expanduser().resolve()
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir = self.output_prefix.parent / f"{self.output_prefix.name}_intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.data_dir}")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >= 0")
        for name, config in GROUND_TRUTHS.items():
            minimum = config["min_coverage"]
            positive = config["positive_threshold"]
            negative = config["negative_threshold"]
            if minimum < 1:
                raise ValueError(f"{name}: min_coverage must be >= 1")
            if not 0 <= negative < positive <= 1:
                raise ValueError(f"{name}: thresholds must satisfy 0 <= negative < positive <= 1")

    def find_optional_unique(self, pattern: str) -> Path | None:
        matches = sorted(self.data_dir.rglob(pattern))
        if len(matches) > 1:
            paths = "\n  ".join(map(str, matches))
            raise RuntimeError(f"File pattern is not unique: {pattern}\n  Matches:\n  {paths}")
        return matches[0] if matches else None

    def find_unique(self, pattern: str) -> Path:
        match = self.find_optional_unique(pattern)
        if match is None:
            raise RuntimeError(f"No file found: {pattern}")
        return match

    def find_replicates(self) -> dict[str, tuple[Path, Path]]:
        result = {}
        for method in METHODS:
            rep1 = self.find_unique(f"GSM5649*{method}*{self.genome}_LAB01_REP01.bedGraph")
            standard_rep2_pattern = f"GSM5649*{method}*{self.genome}_LAB01_REP02.bedGraph"
            rep2 = self.find_optional_unique(standard_rep2_pattern)
            selection_mode = "LAB01_REP01 + LAB01_REP02"

            if rep2 is None and method == "EMSeq":
                fallback_pattern = f"GSM5649*{method}*{self.genome}_LAB02_REP01.bedGraph"
                rep2 = self.find_optional_unique(fallback_pattern)
                if rep2 is not None:
                    selection_mode = "LAB01_REP01 + LAB02_REP01 (EMSeq fallback)"

            if rep2 is None:
                if method == "EMSeq":
                    raise RuntimeError(
                        f"EMSeq is missing a second dataset: neither {standard_rep2_pattern} "
                        f"nor GSM5649*{method}*{self.genome}_LAB02_REP01.bedGraph was found"
                    )
                raise RuntimeError(f"No file found: {standard_rep2_pattern}")

            result[method] = (rep1, rep2)
            print(
                f"{method}:\n  replicate_1={rep1}\n  replicate_2={rep2}\n"
                f"  selection_mode={selection_mode}"
            )
        return result

    def load_counts(self, path: Path) -> dict[tuple[str, int, int], tuple[int, int]]:
        values: dict[tuple[str, int, int], list[int]] = defaultdict(lambda: [0, 0])
        malformed = 0
        scanned = 0
        accepted = 0
        print(f"    Reading: {path.name}", flush=True)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                scanned += 1
                if line.strip() and not line.startswith(("#", "track", "browser")):
                    fields = line.split()
                    if len(fields) < 6:
                        malformed += 1
                    else:
                        try:
                            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                            meth, unmeth = int(float(fields[4])), int(float(fields[5]))
                        except ValueError:
                            malformed += 1
                        else:
                            if AUTOSOME.fullmatch(chrom) and start >= 0 and end > start and meth >= 0 and unmeth >= 0:
                                values[(chrom, start, end)][0] += meth
                                values[(chrom, start, end)][1] += unmeth
                                accepted += 1
                if self.progress_every and scanned % self.progress_every == 0:
                    print(
                        f"      Scanned {scanned:,} lines, retained {accepted:,} autosomal records, "
                        f"current unique sites: {len(values):,}",
                        flush=True,
                    )
        print(
            f"    Read complete: scanned {scanned:,} lines, retained {accepted:,} records, "
            f"unique sites: {len(values):,}, malformed records: {malformed:,}",
            flush=True,
        )
        return {key: (counts[0], counts[1]) for key, counts in values.items()}

    def merge_replicates(self, method: str, paths: tuple[Path, Path]) -> dict[tuple[str, int, int], Counts]:
        rep1, rep2 = (self.load_counts(path) for path in paths)
        common = rep1.keys() & rep2.keys()
        keys = common if self.require_both_replicates else rep1.keys() | rep2.keys()
        merged = {}
        output = self.intermediate_dir / f"{method}_merged.bed"
        with output.open("w", encoding="utf-8") as handle:
            handle.write("chrom\tstart\tend\tmeth_fraction\tcoverage\tmeth_count\tunmeth_count\n")
            for chrom, start, end in sorted(keys):
                m1, u1 = rep1.get((chrom, start, end), (0, 0))
                m2, u2 = rep2.get((chrom, start, end), (0, 0))
                meth, unmeth = m1 + m2, u1 + u2
                coverage = meth + unmeth
                if not coverage:
                    continue
                fraction = meth / coverage
                merged[(chrom, start, end)] = (fraction, coverage, meth, unmeth)
                handle.write(f"{chrom}\t{start}\t{end}\t{fraction:.8f}\t{coverage}\t{meth}\t{unmeth}\n")
        print(
            f"{method}: REP01={len(rep1):,}, REP02={len(rep2):,}, "
            f"common={len(common):,}, merged={len(merged):,}"
        )
        return merged

    def filter_method(
        self,
        method: str,
        merged: dict[tuple[str, int, int], Counts],
        ground_truth_name: str,
        config: dict[str, float | int | bool],
    ):
        positives: set[Site] = set()
        negatives: set[Site] = set()
        details: dict[Site, Counts] = {}
        paths = {
            1: self.intermediate_dir / f"{method}_{ground_truth_name}_positive.bed",
            0: self.intermediate_dir / f"{method}_{ground_truth_name}_negative.bed",
        }
        handles = {label: path.open("w", encoding="utf-8") for label, path in paths.items()}
        header = "chrom\tstart\tend\tmeth_fraction\tcoverage\tmeth_count\tunmeth_count\tstrand\n"
        min_coverage = int(config["min_coverage"])
        positive_threshold = float(config["positive_threshold"])
        negative_threshold = float(config["negative_threshold"])
        negative_inclusive = bool(config["negative_inclusive"])
        try:
            for handle in handles.values():
                handle.write(header)
            for (chrom, start, end), counts in sorted(merged.items()):
                fraction, coverage, meth, unmeth = counts
                if coverage < min_coverage:
                    continue
                negative_match = (
                    fraction <= negative_threshold if negative_inclusive else fraction < negative_threshold
                )
                if fraction >= positive_threshold:
                    label, target = 1, positives
                elif negative_match:
                    label, target = 0, negatives
                else:
                    continue
                for site in ((chrom, start, "+"), (chrom, end - 1, "-")):
                    site_chrom, position, strand = site
                    target.add(site)
                    details[site] = counts
                    handles[label].write(
                        f"{site_chrom}\t{position}\t{position + 1}\t{fraction:.8f}\t"
                        f"{coverage}\t{meth}\t{unmeth}\t{strand}\n"
                    )
        finally:
            for handle in handles.values():
                handle.close()
        negative_operator = "<=" if negative_inclusive else "<"
        print(
            f"{method} [{ground_truth_name}]: coverage >= {min_coverage}, "
            f"positive >= {positive_threshold}, negative {negative_operator} {negative_threshold}; "
            f"positive labels={len(positives):,}, negative labels={len(negatives):,}"
        )
        return positives, negatives, details

    @staticmethod
    def intersect(site_sets: dict[str, set[Site]]) -> set[Site]:
        common = set(site_sets[METHODS[0]])
        for method in METHODS[1:]:
            common &= site_sets[method]
            print(f"  After intersection with {method}: {len(common):,}")
        return common

    def export(
        self,
        ground_truth_name: str,
        config: dict[str, float | int | bool],
        positives: set[Site],
        negatives: set[Site],
        details: dict[str, dict[Site, Counts]],
    ) -> None:
        labels: dict[str, dict[str, dict[int, int]]] = {}
        for label, sites in ((1, positives), (0, negatives)):
            for chrom, position, strand in sites:
                strand_name = "forward" if strand == "+" else "reverse"
                labels.setdefault(chrom, {"forward": {}, "reverse": {}})[strand_name][position] = label

        result_prefix = Path(f"{self.output_prefix}_{ground_truth_name}")
        pickle_path = Path(f"{result_prefix}_filtered_data.pickle")
        with pickle_path.open("wb") as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        bed_path = Path(f"{result_prefix}_final_sites.bed")
        metrics_path = Path(f"{result_prefix}_final_sites_with_metrics.tsv")
        columns = [item for method in METHODS for item in (f"{method}_fraction", f"{method}_coverage")]
        with bed_path.open("w", encoding="utf-8") as bed, metrics_path.open("w", encoding="utf-8") as metrics:
            metrics.write("chrom\tstart\tend\tlabel\tstrand\t" + "\t".join(columns) + "\n")
            for label, sites in ((1, positives), (0, negatives)):
                for chrom, position, strand in sorted(sites):
                    bed.write(f"{chrom}\t{position}\t{position + 1}\t{label}\t{strand}\n")
                    row = []
                    for method in METHODS:
                        fraction, coverage, _, _ = details[method][(chrom, position, strand)]
                        row.extend((f"{fraction:.8f}", str(coverage)))
                    metrics.write(
                        f"{chrom}\t{position}\t{position + 1}\t{label}\t{strand}\t" + "\t".join(row) + "\n"
                    )

        report = Path(f"{result_prefix}_statistics.txt")
        min_coverage = int(config["min_coverage"])
        positive_threshold = float(config["positive_threshold"])
        negative_threshold = float(config["negative_threshold"])
        negative_inclusive = bool(config["negative_inclusive"])
        negative_operator = "<=" if negative_inclusive else "<"
        with report.open("w", encoding="utf-8") as handle:
            handle.write(f"{ground_truth_name} ground-truth filtering statistics\n")
            handle.write(f"genome: {self.genome}\nmin_coverage: {min_coverage}\n")
            handle.write(f"positive_threshold: >= {positive_threshold}\n")
            handle.write(f"negative_threshold: {negative_operator} {negative_threshold}\n")
            handle.write(f"require_both_replicates: {self.require_both_replicates}\n")
            handle.write(f"positive_sites: {len(positives)}\nnegative_sites: {len(negatives)}\n")
            handle.write(f"total_sites: {len(positives) + len(negatives)}\n")
        print(f"Outputs: {pickle_path}\n         {bed_path}\n         {metrics_path}\n         {report}")

    @staticmethod
    def iter_merged_with_coverage(path: Path, min_coverage: int):
        with path.open(encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                fields = line.split()
                if len(fields) < 7:
                    continue
                coverage = int(fields[4])
                if coverage < min_coverage:
                    continue
                key = (fields[0], int(fields[1]), int(fields[2]))
                counts: Counts = (float(fields[3]), coverage, int(fields[5]), int(fields[6]))
                yield key, counts

    def export_correlation(self, min_coverage: int = 10) -> None:
        iterators = {
            method: self.iter_merged_with_coverage(
                self.intermediate_dir / f"{method}_merged.bed", min_coverage
            )
            for method in METHODS
        }
        current = {method: next(iterator, None) for method, iterator in iterators.items()}
        frequencies: dict[str, dict[str, dict[int, float]]] = {}
        result_prefix = Path(f"{self.output_prefix}_correlation")
        bed_path = Path(f"{result_prefix}_final_sites.bed")
        metrics_path = Path(f"{result_prefix}_final_sites_with_metrics.tsv")
        columns = [item for method in METHODS for item in (f"{method}_fraction", f"{method}_coverage")]
        interval_count = 0
        stranded_count = 0

        with bed_path.open("w", encoding="utf-8") as bed, metrics_path.open("w", encoding="utf-8") as metrics:
            metrics.write(
                "chrom\tstart\tend\tstrand\tmean_frequency\t" + "\t".join(columns) + "\n"
            )
            while all(item is not None for item in current.values()):
                target = max(item[0] for item in current.values() if item is not None)
                advanced = False
                for method in METHODS:
                    while current[method] is not None and current[method][0] < target:
                        current[method] = next(iterators[method], None)
                        advanced = True
                if not all(item is not None for item in current.values()):
                    break
                if advanced or any(current[method][0] != target for method in METHODS):
                    continue

                method_counts = {method: current[method][1] for method in METHODS}
                mean_frequency = sum(counts[0] for counts in method_counts.values()) / len(METHODS)
                chrom, start, end = target
                metric_values = []
                for method in METHODS:
                    fraction, coverage, _, _ = method_counts[method]
                    metric_values.extend((f"{fraction:.8f}", str(coverage)))
                metric_suffix = "\t".join(metric_values)

                for position, strand, strand_name in (
                    (start, "+", "forward"),
                    (end - 1, "-", "reverse"),
                ):
                    frequencies.setdefault(chrom, {"forward": {}, "reverse": {}})[strand_name][position] = mean_frequency
                    bed.write(
                        f"{chrom}\t{position}\t{position + 1}\t{mean_frequency:.8f}\t{strand}\n"
                    )
                    metrics.write(
                        f"{chrom}\t{position}\t{position + 1}\t{strand}\t"
                        f"{mean_frequency:.8f}\t{metric_suffix}\n"
                    )
                    stranded_count += 1

                interval_count += 1
                if self.progress_every and interval_count % self.progress_every == 0:
                    print(
                        f"  [Correlation] Emitted {interval_count:,} CpG intervals and "
                        f"{stranded_count:,} stranded sites",
                        flush=True,
                    )
                for method in METHODS:
                    current[method] = next(iterators[method], None)

        pickle_path = Path(f"{result_prefix}_frequency.pickle")
        with pickle_path.open("wb") as handle:
            pickle.dump(frequencies, handle, protocol=pickle.HIGHEST_PROTOCOL)

        report = Path(f"{result_prefix}_statistics.txt")
        with report.open("w", encoding="utf-8") as handle:
            handle.write("correlation ground-truth statistics\n")
            handle.write(f"genome: {self.genome}\n")
            handle.write(f"min_coverage_per_method: {min_coverage}\n")
            handle.write("frequency_filter: none\n")
            handle.write("reference_frequency: arithmetic mean of six method frequencies\n")
            handle.write(f"require_both_replicates: {self.require_both_replicates}\n")
            handle.write(f"cpg_intervals: {interval_count}\n")
            handle.write(f"stranded_sites: {stranded_count}\n")
        print(f"Outputs: {pickle_path}\n         {bed_path}\n         {metrics_path}\n         {report}")

    def run(self) -> None:
        self.validate()
        print("The following ground-truth datasets will be generated:")
        for name, config in GROUND_TRUTHS.items():
            operator = "<=" if config["negative_inclusive"] else "<"
            print(
                f"  {name}: coverage >= {config['min_coverage']}, "
                f"positive >= {config['positive_threshold']}, "
                f"negative {operator} {config['negative_threshold']}"
            )
        print(
            "  correlation: coverage >= 10 for each method, no frequency filtering, "
            "reference frequency is the arithmetic mean across six methods"
        )

        files = self.find_replicates()
        results = {
            name: {"positive_sets": {}, "negative_sets": {}, "details": {}}
            for name in GROUND_TRUTHS
        }
        for method_index, method in enumerate(METHODS, 1):
            print(f"\n[Method {method_index}/{len(METHODS)}] {method}", flush=True)
            merged = self.merge_replicates(method, files[method])
            for ground_truth_index, (name, config) in enumerate(GROUND_TRUTHS.items(), 1):
                print(
                    f"  [Ground truth {ground_truth_index}/{len(GROUND_TRUTHS)}] {name}",
                    flush=True,
                )
                positives, negatives, details = self.filter_method(method, merged, name, config)
                results[name]["positive_sets"][method] = positives
                results[name]["negative_sets"][method] = negatives
                results[name]["details"][method] = details

        for name, config in GROUND_TRUTHS.items():
            result = results[name]
            print(f"{name} positive-label intersection across six methods:")
            positives = self.intersect(result["positive_sets"])
            print(f"{name} negative-label intersection across six methods:")
            negatives = self.intersect(result["negative_sets"])
            self.export(name, config, positives, negatives, result["details"])

        print("Correlation coverage intersection and frequency mean across six methods:")
        self.export_correlation(min_coverage=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge EpiQC replicates and generate ModFormer ground truth"
    )
    parser.add_argument("data_dir", help="Root directory recursively searched for bedGraph files")
    parser.add_argument("--genome", default="HG002", help="Sample name (default: HG002)")
    parser.add_argument("--output-prefix", default="filtered_methylation")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Report read progress after this many lines (default: 1000000; 0 disables reporting)",
    )
    parser.add_argument(
        "--allow-single-replicate-sites",
        action="store_true",
        help="Allow sites present in only one replicate; by default both replicates must cover the site",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    StrictMethylationFilter(args).run()


if __name__ == "__main__":
    main()
