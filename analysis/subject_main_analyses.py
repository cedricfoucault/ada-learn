"""
Make main plots for the given subject data file.
- Summary table 
- Performance over sessions
- Learning rate around change points
- Learning rate modulation by variables
- Individual sequences
"""

import argparse
import data_analysis_utils as dana
import subject_summary_table
import performance_over_sessions_subject
import learning_rate_around_cp_subject
import learning_rate_relationship_cpp_uncertainty_subject
import subject_sessions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--is_group_file", action="store_true", default=False)
    args = parser.parse_args()
    file = args.file

    if args.is_group_file:
        data_files = dana.get_subject_data_files_from_group_def(file)
    else:
        data_files = [file]
    
    for data_file in data_files:
        subject_summary_table.run_with_args(data_file)
        performance_over_sessions_subject.run_with_args(data_file)
        learning_rate_around_cp_subject.run_with_args(data_file)
        learning_rate_relationship_cpp_uncertainty_subject.run_with_args(
            data_file)
        subject_sessions.run_with_args(data_file)
