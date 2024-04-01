"""
Script to aggregate data from multiple raw demographics data files obtained
from Prolific.
"""

import pandas as pd

RAW_DEMOGRAPHICS_DATA_FILES = [
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63d95c5a78080dfe9805f20f.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63d95f80f4cc7b1b41dc978f.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63dccbfa4325f63630e93414.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63dcccfb0203be81afa1026a.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63e7ddd8db276ad46a42a291.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63e7dde8ad75f35697ff0cdd.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63ebe1a3178121b385ab86c2.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63ebe26f463910c503e037a9.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63ee6ac609643e895a98abc5.csv",
    "data/demographic-data/ada-learn_23-02-17/prolific_export_63ee6accfa2c99defa333ee7.csv",
]
OUTPUT_DEMOGRAPHICS_DATA_FILE = "data/demographic-data/ada-learn_23-02-17/ada-learn_23-02-17_demographics_data.csv"


def main():
    dfs = [pd.read_csv(f) for f in RAW_DEMOGRAPHICS_DATA_FILES]
    df = pd.concat(dfs)
    df.to_csv(OUTPUT_DEMOGRAPHICS_DATA_FILE)

if __name__ == '__main__':
    main()