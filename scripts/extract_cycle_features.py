#!/usr/bin/env python3
# Fill tbl_cycle_summary and tbl_cell_features from the stored .mpr files.
# Re-running replaces a file's rows, so run it again after the feature code changes.
#   python3 -m scripts.extract_cycle_features --all
#   python3 -m scripts.extract_cycle_features --record CELL-0001

import argparse

from services.cycle_features import analyse_and_save, cycling_files


def main(argv=None):
    parser = argparse.ArgumentParser()
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every stored file")
    which.add_argument("--record", help="only this record's files")
    args = parser.parse_args(argv)

    files = cycling_files(None if args.all else args.record)
    if not files:
        print("No cycling files found.")
        return
    for cycling_id, record_id, file_path in files:
        n_cycles, kept, result = analyse_and_save(cycling_id, record_id, file_path)
        print(
            f"{record_id} (file {cycling_id}): {n_cycles} cycles, "
            f"{kept} passed the gate; {result}"
        )


if __name__ == "__main__":
    main()
