"""HPOA -> Phase5 LIRICAL background frequency table.

Input : phenotype.hpoa (HPO Consortium, 2026-02-16 release)
Output: hpo_background_freq.json — {hp_id: background_freq (0..1), ...}

Method (LIRICAL Robinson 2020 spirit, simplified):
  freq_in_background(HP_i) = (sum_{D} f(HP_i | D)) / N_diseases

  - f(HP_i | D) is the per-disease frequency parsed from HPOA's "frequency"
    column (handles "n/m", "%", HP:00402xx categorical, empty=default).
  - N_diseases is the count of all unique disease_id annotated in HPOA.
  - Laplace-style smoothing: floor at 1 / (2 * N_diseases) so unseen-but-
    plausible terms do not blow up LR = freq(HP|D) / 0.

Output also writes a CSV for quick eyeballing.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HPOA_PATH = Path(__file__).parent / "phenotype.hpoa"
JSON_OUT = Path(__file__).parent / "hpo_background_freq.json"
CSV_OUT = Path(__file__).parent / "hpo_background_freq.csv"
META_OUT = Path(__file__).parent / "hpo_background_freq.meta.json"

# HPO categorical frequency terms (HPO ontology HP:0040279 subtree)
HPO_FREQ_TERMS = {
    "HP:0040280": 1.00,   # Obligate
    "HP:0040281": 0.90,   # Very frequent (80-99%)
    "HP:0040282": 0.55,   # Frequent (30-79%)
    "HP:0040283": 0.17,   # Occasional (5-29%)
    "HP:0040284": 0.025,  # Very rare (<5% but >0)
    "HP:0040285": 0.0,    # Excluded
}

DEFAULT_FREQ_IF_MISSING = 0.5  # LIRICAL default when annotation has no frequency


def parse_frequency(raw: str) -> float | None:
    """Parse HPOA 'frequency' field → float in [0,1].

    Examples:
        ""           → None  (caller substitutes default)
        "1/2"        → 0.5
        "3 of 14"    → 0.214
        "30%"        → 0.30
        "HP:0040282" → 0.55
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    # HPO categorical
    if s.startswith("HP:"):
        return HPO_FREQ_TERMS.get(s)

    # Percentage
    m = re.match(r"^\s*([\d.]+)\s*%\s*$", s)
    if m:
        try:
            return min(max(float(m.group(1)) / 100.0, 0.0), 1.0)
        except ValueError:
            return None

    # n/m or "n of m"
    m = re.match(r"^\s*(\d+)\s*[/of]+\s*(\d+)\s*$", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den > 0:
            return min(max(num / den, 0.0), 1.0)

    return None


def main() -> int:
    if not HPOA_PATH.exists():
        print(f"ERROR: {HPOA_PATH} not found", file=sys.stderr)
        return 1

    # accumulator: hp_id -> sum of per-disease frequencies
    hp_freq_sum: dict[str, float] = defaultdict(float)
    # set of disease ids seen
    diseases: set[str] = set()
    # diagnostic counters
    rows_total = 0
    rows_with_freq = 0
    rows_excluded = 0  # qualifier == 'NOT' (negated)
    aspect_counts: dict[str, int] = defaultdict(int)
    version_line = ""

    with HPOA_PATH.open(encoding="utf-8", newline="") as fh:
        # Skip leading comment lines but capture version
        header_line = None
        for line in fh:
            if line.startswith("#"):
                if "version:" in line and not version_line:
                    version_line = line.strip().lstrip("#").strip()
                continue
            header_line = line.rstrip("\n")
            break

        if header_line is None:
            print("ERROR: no header found", file=sys.stderr)
            return 1

        fieldnames = header_line.split("\t")
        reader = csv.DictReader(fh, fieldnames=fieldnames, delimiter="\t")

        for row in reader:
            rows_total += 1
            qualifier = (row.get("qualifier") or "").strip().upper()
            aspect = (row.get("aspect") or "").strip()
            aspect_counts[aspect] += 1

            # Only count Phenotypic abnormality (aspect == 'P').
            # Skip C (Clinical course), I (Inheritance), M (Modifier), etc.
            if aspect != "P":
                continue
            # Negated annotations are not background presence
            if qualifier == "NOT":
                rows_excluded += 1
                continue

            hp_id = (row.get("hpo_id") or "").strip()
            disease_id = (row.get("database_id") or "").strip()
            if not hp_id.startswith("HP:") or not disease_id:
                continue

            freq_raw = row.get("frequency") or ""
            f = parse_frequency(freq_raw)
            if f is None:
                f = DEFAULT_FREQ_IF_MISSING
            else:
                rows_with_freq += 1

            hp_freq_sum[hp_id] += f
            diseases.add(disease_id)

    n_disease = len(diseases)
    n_hp = len(hp_freq_sum)
    floor = 1.0 / (2.0 * n_disease) if n_disease else 1e-6

    background: dict[str, float] = {}
    for hp_id, total in hp_freq_sum.items():
        bg = total / n_disease if n_disease else 0.0
        background[hp_id] = max(bg, floor)

    # write JSON (sorted by HP ID for stable diffs)
    with JSON_OUT.open("w", encoding="utf-8") as fh:
        json.dump(
            {k: round(v, 8) for k, v in sorted(background.items())},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    # write CSV
    with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hp_id", "background_freq", "sum_freq_across_diseases"])
        for hp_id in sorted(background):
            w.writerow([hp_id, f"{background[hp_id]:.8f}", f"{hp_freq_sum[hp_id]:.6f}"])

    meta = {
        "source": "HPO Consortium phenotype.hpoa",
        "source_version": version_line,
        "n_disease_total": n_disease,
        "n_hpo_terms_with_annotation": n_hp,
        "rows_total": rows_total,
        "rows_with_explicit_frequency": rows_with_freq,
        "rows_negated_skipped": rows_excluded,
        "aspect_distribution": dict(aspect_counts),
        "smoothing_floor": floor,
        "method": (
            "background_freq(HP) = sum over diseases D of f(HP|D) / N_diseases. "
            "Per-disease frequency f parsed from HPOA 'frequency' column: "
            "n/m fractions, %, HP:00402xx categorical (Obligate=1.0, "
            "Very frequent=0.9, Frequent=0.55, Occasional=0.17, Very rare=0.025, "
            "Excluded=0). Missing frequency -> 0.5 (LIRICAL default). "
            "Laplace-style floor = 1 / (2 * N_diseases)."
        ),
        "intended_use": (
            "Phase 5 LIRICAL-style scoring denominator. Per Robinson PN et al. "
            "Am J Hum Genet 2020;107(3):403-417 (PMID:32755546). "
            "LR(HP|D) = freq_in_disease(HP|D) / background_freq(HP)."
        ),
    }
    with META_OUT.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"OK: {n_hp} HPO terms across {n_disease} diseases")
    print(f"     rows_total={rows_total}, with_freq={rows_with_freq}, "
          f"negated_skipped={rows_excluded}")
    print(f"     floor={floor:.6g}")
    print(f"     -> {JSON_OUT.name}")
    print(f"     -> {CSV_OUT.name}")
    print(f"     -> {META_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
