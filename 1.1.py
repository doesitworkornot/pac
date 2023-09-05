import argparse
import random

parser = argparse.ArgumentParser(description='yeah')
parser.add_argument('-n')
args = parser.parse_args()
n = int(args.n)
print(n)

arr = [random.random() for i in range(n)]
print(arr)
for i in range(n):
    for j in range(n - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
print(arr)
