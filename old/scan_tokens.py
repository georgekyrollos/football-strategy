# scan_tokens.py
from read_matrix import load_pro_style_chart_csv
from transition import parse_cell

chart = load_pro_style_chart_csv("Football Strategy Pro Style.csv")

tokens = sorted(set(chart.values.ravel()))
bad = []

for t in tokens:
    try:
        parse_cell(t)
    except Exception as e:
        bad.append((t, repr(e)))

print(f"Unique tokens: {len(tokens)}")
print(f"Unparsed tokens: {len(bad)}")
for t, e in bad:
    print(t, "->", e)
