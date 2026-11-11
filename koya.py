
import os
import re
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Gait Dataset\390- Koya University Smartphone Gait Dataset\Dataset\Dataset\smartphone-gait-biometric-390\All_Sensor_390")

# Output is outside All_Sensor_390 so it cannot be mistaken for a participant.
OUT = ROOT.parent / "Koya_Gait_Combined_12Feature_Output"

EXPECTED_PARTICIPANTS = 390
EXPECTED_SESSIONS_PER_PARTICIPANT = 10

# Dataset paper
FS = 30.0
LOWPASS_HZ = 10.0
FILTER_ORDER = 4

# Remove edge samples before filtering/peak detection.
TRIM_START = 30
TRIM_END = 60

# Peak detection parameters from the paper.
PEAK_DISTANCE = 10
PEAK_HEIGHT = 10.0
PEAK_PROMINENCE = 2.5
MAX_INTERPEAK = 22

# Biomechanically valid full-cycle duration.
MIN_CYCLE_LEN = 28
MAX_CYCLE_LEN = 44

# Final feature set: exactly 12 sensor features.
FEATURES = [
    "accx", "accy", "accz", "acc_mag",
    "gyx", "gyy", "gyz", "gy_mag",
    "magx", "magy", "magz", "mag_mag"
]

RAW_COLS = [
    "accx", "accy", "accz",
    "magx", "magy", "magz",
    "gyx", "gyy", "gyz"
]


# ============================================================
# 2. HELPERS
# ============================================================

def natural_key(s):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(s))
    ]


def find_csvs(folder):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
    return sorted(files, key=lambda x: natural_key(x.name))


def read_sensor_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


def add_magnitudes(df):
    a = df[["accx", "accy", "accz"]].to_numpy(float)
    g = df[["gyx", "gyy", "gyz"]].to_numpy(float)
    m = df[["magx", "magy", "magz"]].to_numpy(float)

    df = df.copy()
    df["acc_mag"] = np.linalg.norm(a, axis=1)
    df["gy_mag"] = np.linalg.norm(g, axis=1)
    df["mag_mag"] = np.linalg.norm(m, axis=1)
    return df


def lowpass(x):
    nyq = FS / 2.0
    b, a = butter(
        FILTER_ORDER,
        LOWPASS_HZ / nyq,
        btype="low"
    )
    return filtfilt(b, a, x)


def get_cycle_bounds(acc_mag):
    """
    Paper-based gait-cycle extraction:

    1. Trim 30 samples from the beginning and 60 from the end.
    2. Fourth-order Butterworth low-pass filter at 10 Hz.
    3. Detect ACC-magnitude peaks:
         distance >= 10 samples
         height >= 10 m/s
         prominence >= 2.5 m/s
    4. Split peaks when consecutive peaks are >22 samples apart.
    5. Use triplets of consecutive peaks.
    6. Keep cycles whose first-to-third duration is 28..44 samples.

    The detailed Section 3.2 processing description uses the
    first-to-third peak span as one complete gait cycle.
    """

    n = len(acc_mag)

    if n <= TRIM_START + TRIM_END + 30:
        return []

    acc = acc_mag[TRIM_START:n - TRIM_END]

    if len(acc) < 30:
        return []

    acc_f = lowpass(acc)

    peaks, _ = find_peaks(
        acc_f,
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
        prominence=PEAK_PROMINENCE
    )

    if len(peaks) < 3:
        return []

    # Keep contiguous peak groups in which each step interval <= 22 samples.
    groups = []
    current = [int(peaks[0])]

    for p in peaks[1:]:
        p = int(p)

        if p - current[-1] <= MAX_INTERPEAK:
            current.append(p)
        else:
            if len(current) >= 3:
                groups.append(current)
            current = [p]

    if len(current) >= 3:
        groups.append(current)

    cycles = []

    # Full stride = first peak -> third peak.
    # Sliding triplets are used because Section 3.2 describes
    # triplets of consecutive refined peaks.
    for group in groups:
        for i in range(len(group) - 2):
            s = group[i]
            e = group[i + 2]
            length = e - s

            if MIN_CYCLE_LEN <= length <= MAX_CYCLE_LEN:
                # Convert trimmed-signal indices back to original-file indices.
                cycles.append((
                    s + TRIM_START,
                    e + TRIM_START
                ))

    return cycles


def extract_cycles(df):
    df = add_magnitudes(df)

    bounds = get_cycle_bounds(
        df["acc_mag"].to_numpy(float)
    )

    if not bounds:
        return []

    arr = df[FEATURES].to_numpy(float)

    cycles = []

    for s, e in bounds:
        if 0 <= s < e <= len(arr):
            x = arr[s:e].copy()

            if MIN_CYCLE_LEN <= len(x) <= MAX_CYCLE_LEN:
                cycles.append(x)

    return cycles


def paa_resample(x, target_len):
    """
    Fractional-overlap Piecewise Aggregate Approximation.

    It supports both downsampling and upsampling, which is useful
    because valid gait cycles range from 28 to 44 samples while the
    final global target is determined from the dataset median.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x[:, None]

    n = len(x)

    if n == 0:
        return np.empty((0, x.shape[1]))

    if n == target_len:
        return x.copy()

    out = np.empty(
        (target_len, x.shape[1]),
        dtype=float
    )

    for j in range(target_len):
        left = j * n / target_len
        right = (j + 1) * n / target_len

        i0 = max(0, int(math.floor(left)))
        i1 = min(n - 1, int(math.ceil(right) - 1))

        total = np.zeros(x.shape[1], dtype=float)
        weight = 0.0

        for i in range(i0, i1 + 1):
            overlap = max(
                0.0,
                min(right, i + 1) - max(left, i)
            )

            if overlap > 0:
                total += x[i] * overlap
                weight += overlap

        if weight > 0:
            out[j] = total / weight
        else:
            out[j] = x[min(n - 1, int(left))]

    return out


def get_participants(root):
    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]

    dirs = sorted(
        dirs,
        key=lambda x: natural_key(x.name)
    )

    # A participant directory must contain at least one CSV.
    dirs = [
        p for p in dirs
        if len(find_csvs(p)) > 0
    ]

    return dirs


def write_csv_header(path):
    pd.DataFrame(
        columns=FEATURES + ["Label"]
    ).to_csv(path, index=False)


# ============================================================
# 3. MAIN
# ============================================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(
            f"\nDataset folder not found:\n{ROOT}\n\n"
            "Check the ROOT path at the top of this script."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    participant_dirs = get_participants(ROOT)

    print(
        f"Participant folders found: "
        f"{len(participant_dirs)}"
    )

    # Stop instead of silently producing a wrong 390-class dataset.
    if len(participant_dirs) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            f"\nExpected {EXPECTED_PARTICIPANTS} participant folders, "
            f"but found {len(participant_dirs)}.\n"
            "The script stopped to prevent building an incorrect dataset."
        )

    # Stable numeric labels:
    # sorted participant folder #1 -> Label 1
    # sorted participant folder #2 -> Label 2
    # ...
    # sorted participant folder #390 -> Label 390
    label_map = {
        p.name: i + 1
        for i, p in enumerate(participant_dirs)
    }

    # Save mapping because the main ML CSV intentionally contains
    # only the requested 12 features + Label.
    mapping = pd.DataFrame([
        {
            "Participant_ID": p.name,
            "Label": label_map[p.name]
        }
        for p in participant_dirs
    ])

    mapping.to_csv(
        OUT / "participant_label_map.csv",
        index=False
    )

    # --------------------------------------------------------
    # Build recording list
    # --------------------------------------------------------

    file_records = []
    total_files = 0

    for pdir in participant_dirs:

        files = find_csvs(pdir)

        if len(files) != EXPECTED_SESSIONS_PER_PARTICIPANT:
            warnings.warn(
                f"{pdir.name}: expected "
                f"{EXPECTED_SESSIONS_PER_PARTICIPANT} sessions, "
                f"found {len(files)}"
            )

        for session_no, f in enumerate(files, start=1):

            file_records.append({
                "participant": pdir.name,
                "label": label_map[pdir.name],
                "session": session_no,
                "path": f
            })

        total_files += len(files)

    print(
        f"CSV recordings found: {total_files}"
    )

    # --------------------------------------------------------
    # PASS 1
    # Find all valid gait-cycle lengths.
    # This lets us choose ONE global target length.
    # --------------------------------------------------------

    print(
        "\nPASS 1/2: extracting valid cycle lengths..."
    )

    lengths = []
    file_stats = []

    for idx, rec in enumerate(
        file_records,
        start=1
    ):

        try:

            df = read_sensor_csv(
                rec["path"]
            )

            cycles = extract_cycles(df)

            lens = [
                len(c)
                for c in cycles
            ]

            lengths.extend(lens)

            file_stats.append({
                "Participant_ID": rec["participant"],
                "Label": rec["label"],
                "Session": rec["session"],
                "File": str(rec["path"]),
                "Samples": len(df),
                "Valid_Cycles": len(cycles),
                "Status": "OK"
            })

        except Exception as e:

            file_stats.append({
                "Participant_ID": rec["participant"],
                "Label": rec["label"],
                "Session": rec["session"],
                "File": str(rec["path"]),
                "Samples": 0,
                "Valid_Cycles": 0,
                "Status": f"ERROR: {e}"
            })

        if idx % 100 == 0 or idx == len(file_records):
            print(
                f"  processed "
                f"{idx}/{len(file_records)}"
            )

    if not lengths:
        raise RuntimeError(
            "\nNo valid gait cycles were extracted.\n"
            "Check sensor columns, units, or peak-detection parameters."
        )

    # Global target = median valid cycle length.
    # It is constrained to the paper's valid 28..44 range.
    global_target = int(
        round(float(np.median(lengths)))
    )

    global_target = max(
        MIN_CYCLE_LEN,
        min(MAX_CYCLE_LEN, global_target)
    )

    print(
        f"\nValid gait cycles found: "
        f"{len(lengths)}"
    )

    print(
        f"Median original cycle length: "
        f"{np.median(lengths):.2f}"
    )

    print(
        f"GLOBAL PAA target length: "
        f"{global_target}"
    )

    pd.DataFrame(file_stats).to_csv(
        OUT / "processing_report.csv",
        index=False
    )

    # --------------------------------------------------------
    # PASS 2
    # PAA-normalize each gait cycle and write one CSV.
    # --------------------------------------------------------

    print(
        "\nPASS 2/2: creating final combined dataset..."
    )

    final_csv = (
        OUT /
        "Koya_Gait_Combined_12Features.csv"
    )

    index_csv = (
        OUT /
        "Koya_Gait_Cycle_Index.csv"
    )

    # Main CSV = exactly 13 columns:
    # 12 features + Label
    write_csv_header(final_csv)

    index_rows = []

    # Current data-row number in the main CSV.
    current_row = 1

    total_cycles_written = 0

    for idx, rec in enumerate(
        file_records,
        start=1
    ):

        try:

            df = read_sensor_csv(
                rec["path"]
            )

            cycles = extract_cycles(df)

            for cycle_no, cycle in enumerate(
                cycles,
                start=1
            ):

                # Global temporal normalization.
                x = paa_resample(
                    cycle,
                    global_target
                )

                out = pd.DataFrame(
                    x,
                    columns=FEATURES
                )

                # Same participant label is repeated for
                # every timestep belonging to that participant.
                out["Label"] = rec["label"]

                out.to_csv(
                    final_csv,
                    mode="a",
                    header=False,
                    index=False
                )

                start_row = current_row
                end_row = (
                    current_row +
                    global_target -
                    1
                )

                index_rows.append({
                    "Participant_ID": rec["participant"],
                    "Label": rec["label"],
                    "Session": rec["session"],
                    "Cycle": cycle_no,
                    "Source_File": str(rec["path"]),
                    "Original_Cycle_Length": len(cycle),
                    "PAA_Length": global_target,
                    "Start_Row": start_row,
                    "End_Row": end_row
                })

                current_row = end_row + 1
                total_cycles_written += 1

        except Exception:
            # The detailed processing_report.csv contains
            # information about files that failed in PASS 1.
            pass

        if idx % 100 == 0 or idx == len(file_records):
            print(
                f"  written from "
                f"{idx}/{len(file_records)} recordings"
            )

    pd.DataFrame(index_rows).to_csv(
        index_csv,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "participants": len(participant_dirs),
        "recordings_found": total_files,
        "valid_gait_cycles": total_cycles_written,
        "target_timesteps_per_cycle": global_target,
        "feature_columns": FEATURES,
        "label_values": "1..390",
        "final_columns": FEATURES + ["Label"],
        "sampling_rate_hz": FS,
        "lowpass_hz": LOWPASS_HZ,
        "filter_order": FILTER_ORDER,
        "trim_start_samples": TRIM_START,
        "trim_end_samples": TRIM_END,
        "peak_distance_samples": PEAK_DISTANCE,
        "peak_height": PEAK_HEIGHT,
        "peak_prominence": PEAK_PROMINENCE,
        "max_interpeak_samples": MAX_INTERPEAK,
        "valid_cycle_length_min": MIN_CYCLE_LEN,
        "valid_cycle_length_max": MAX_CYCLE_LEN,
        "normalization": "Global PAA"
    }

    with open(
        OUT / "dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n==============================================")
    print("DONE")
    print("==============================================")
    print(f"Final dataset : {final_csv}")
    print(f"Cycle index   : {index_csv}")
    print(
        f"Label mapping : "
        f"{OUT / 'participant_label_map.csv'}"
    )
    print(
        f"Report        : "
        f"{OUT / 'processing_report.csv'}"
    )
    print(
        f"Summary       : "
        f"{OUT / 'dataset_summary.json'}"
    )
    print(f"Participants  : {len(participant_dirs)}")
    print(f"Recordings    : {total_files}")
    print(f"Gait cycles   : {total_cycles_written}")
    print(f"Rows/cycle    : {global_target}")
    print(
        f"Columns       : "
        f"{len(FEATURES) + 1} "
        f"(12 features + Label)"
    )
    print("==============================================")


if __name__ == "__main__":
    main()
