import numpy as np

print('Task 1')
a = np.array([1, 2, 3, 4, 5, 6, 5, 6, 5, 6, 4, 5, 6, 6, 6])
u, count = np.unique(a, return_counts=True)
g_ind = np.argsort(-count)
print(a[g_ind])

print('\nTask 2')
m = 10
n = 10
a = np.random.randint(0, 255, size=(m, n))
print(len(np.unique(a)))

print('\nTask 3')
def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

a = np.random.randint(0, 255, size=10)
n = 3
print(a)
print(moving_avg(a, n))

print('\nTask 4')
m = 10
n = 3
a = np.random.randint(0, 15, size=(m, n))

def is_good(a):
    return (a[0] + a[1] > a[2] and a[1] + a[2] > a[0] and a[0] + a[2] > a[1])

print(a[np.apply_along_axis(is_good, 1, a)])

