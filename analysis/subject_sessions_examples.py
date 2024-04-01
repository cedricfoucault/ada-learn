"""
Plot an example subject session for each task. 
"""

import subject_sessions

example_ada_pos_data_file = "data/ada-pos_subject-60290c574980dd7f8e378b84_run-63a9838a208f5a27774db1a4_2022-12-26.csv"
example_ada_pos_isess = 0
example_ada_prob_data_file = "data/ada-prob_subject-582ee44408b46b00010dbb07_run-63c01ead8733d17ea5eea8aa_2023-01-12.csv"
example_ada_prob_isess = 11

if __name__ == '__main__':
    subject_sessions.run_with_args(example_ada_pos_data_file,
        i_sessions=[example_ada_pos_isess],
        with_generative=True,
        exts=["pdf"])
    subject_sessions.run_with_args(example_ada_prob_data_file,
        i_sessions=[example_ada_prob_isess],
        with_generative=True,
        exts=["pdf"])
