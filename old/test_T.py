from read_matrix import load_pro_style_chart_csv
from transition import State, successors

chart = load_pro_style_chart_csv("Football Strategy Pro Style.csv")

s = State(x=20, down=1, to_first=10)
print("Cell F vs 4:", chart.loc["F", "4"])

for tr in successors(chart, s, off_play=4, def_play="F"):
    print(tr)
