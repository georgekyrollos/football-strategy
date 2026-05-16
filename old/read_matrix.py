# read_matrix.py
import pandas as pd

def load_pro_style_chart_csv(csv_path="Football Strategy Pro Style.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={df.columns[0]: "DEF"}).set_index("DEF")
    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = [str(i) for i in range(1, 21)]
    df = df[expected_cols]

    def clean(v):
        if pd.isna(v):
            return ""
        return str(v).strip()

    # apply map per column (no applymap warning)
    chart = df.apply(lambda col: col.map(clean))

    expected_rows = list("ABCDEFGHIJ")
    missing = [r for r in expected_rows if r not in chart.index]
    if missing:
        raise ValueError(f"Missing defense rows: {missing}")

    return chart


if __name__ == "__main__":
    chart = load_pro_style_chart_csv("Football Strategy Pro Style.csv")
    print("Chart shape:", chart.shape)   # should be (10, 20)
    print(chart.head())
    print("F vs 4 =", chart.loc["F", "4"])
