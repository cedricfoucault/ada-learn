"""
Analyze participant demographics data.
"""

import pandas as pd

DEMOGRAPHICS_DATA_FILE = "data/demographic-data/ada-learn_23-02-17/ada-learn_23-02-17_demographics_data.csv"
DEMOGRAPHICS_SUMMARY_FILE = "data/demographic-data/ada-learn_23-02-17/ada-learn_23-02-17_demographics_summary.csv"

def main():
    data = pd.read_csv(DEMOGRAPHICS_DATA_FILE)
    summary_data = {
        "Num. female": (data["Sex"] == "Female").sum(),
        "Num. male": (data["Sex"] == "Male").sum(),
        "Age median": data["Age"].median(),
        "Age interquartile range": (data["Age"].quantile(1/4), data["Age"].quantile(3/4)),
        "Age mean": data["Age"].mean(),
        "Age sd": data["Age"].std(),
        "Num. different countries of birth": len((data["Country of birth"]).unique()),
        "Num. different countries of residence": len((data["Country of residence"]).unique()),
        "Num. different nationalities": len((data["Nationality"]).unique()),
    }
    print(summary_data)
    pd.Series(summary_data).to_csv(DEMOGRAPHICS_SUMMARY_FILE)

if __name__ == '__main__':
    main()