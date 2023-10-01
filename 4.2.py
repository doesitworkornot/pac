import sys
import numpy as np


def read_from(path):
    try:
        with open(path, 'r') as file:
            content = file.read()
        return content.split()
    except FileNotFoundError:
        print(f"File not found: {path}")
    except PermissionError:
        print(f"Permission denied to access: {path}")


def main():
    if len(sys.argv) == 4:
        a = np.array(read_from(sys.argv[1]))
        b = np.array(read_from(sys.argv[2]))
        try:
            p = float(sys.argv[3])
        except ValueError:
            print("Bad p arg")
        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if not len(a) == len(b):
            raise ValueError("Arrays should be the same length")
        print(a, '\n',  b)
        print(f'Probability is {p}')
        ans = np.where(np.random.uniform(0, 1, size=len(a)) > p, a, b)
        print(ans)

    else:
        print("Not enough arguments")


if __name__ == "__main__":
    main()
