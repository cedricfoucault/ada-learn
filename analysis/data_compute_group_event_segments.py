"""
Compute time series of event segments (with start time and end time),
for all data files in the provided group and save it into a separate
with one event segment file for each of the original data file.

These segments include:
- trial events (data: index of the trial)
- slider movement events (data: value of the estimate at start time and end time,
                          trajectory of the estimates during the movement)

The slider movements are segmented from the instantantaneous mousemove events
using the defined maximum time threshold.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import plot_utils as plut

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH / 2, plut.A4_PAPER_CONTENT_WIDTH / 2)

SMALL_BIN_WIDTH_MS = 10
SOA_MS = 1500
BIN_EDGES_MS = [SMALL_BIN_WIDTH_MS * i for i in range(SOA_MS // SMALL_BIN_WIDTH_MS + 1)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_def_file")
    args = parser.parse_args()
    group_def_file = args.group_def_file

    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    n_subjects = len(data_files)

    # Compute time intervals between all slider move events for all subjects in all sessions
    slidermove_time_intervals = []
    for data_file in data_files:
        data = dana.load_data(data_file)
        sessions = data["taskSessionDataArray"]
        sessions_event_segments = []
        for session in sessions:
            event_segments = dana.compute_event_segments(session)
            sessions_event_segments += [event_segments]
        out_file = dana.get_sessions_event_segments_file_for_data_file(data_file,
            create_dir_if_needed=True)
        dana.save_sessions_event_segments(sessions_event_segments, out_file)
        print(f"Sessions event segments saved at {out_file}")
