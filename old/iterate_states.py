from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    x: int        # yards from own goal line, 1..99
    down: int     # 1..4
    to_first: int # 1..10 (exact, for Markov property)

def iter_drive_states(x_min=1, x_max=99, to_first_max=10):
    for x in range(x_min, x_max + 1):
        for down in range(1, 5):
            for to_first in range(1, to_first_max + 1):
                yield State(x=x, down=down, to_first=to_first)

def main():
    n = 0
    for s in iter_drive_states():
        print(s)
        n += 1
    print(f"\nTotal drive-level states: {n}")

if __name__ == "__main__":
    main()
