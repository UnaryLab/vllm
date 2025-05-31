#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd


def rocm_tensor_util(fn):
    df = pd.read_csv(fn)

    # MI300x specific
    n_xcd = 8
    n_cu = 304

    gpu_cycles = df.loc[
        df['Counter_Name'] == 'GRBM_GUI_ACTIVE',
        'Counter_Value'
    ].sum()

    tensor_cycles = df.loc[
        df['Counter_Name'] == 'SQ_VALU_MFMA_BUSY_CYCLES',
        'Counter_Value'
    ].sum()

    tensor_util = (
        100 * tensor_cycles /
        (n_cu * gpu_cycles / n_xcd * 4)
    )

    return tensor_util


def main(fn, rocm):
    if rocm:
        print(rocm_tensor_util(fn))
    else:
        raise ValueError(f"Only rocm is supported")
    
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(prog="genie_duration")
    parser.add_argument(
        "perf_counter_csv",
        type=str,
        help="Filename of performance counters CSV"
    )
    parser.add_argument(
        "--rocm",
        "-r",
        action="store_true",
        help="Pass if using rocm instead of nvidia"
    )
    args = parser.parse_args()
    exit(main(args.perf_counter_csv, args.rocm))

