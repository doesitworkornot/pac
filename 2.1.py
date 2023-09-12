#
# print('Task 1')
# a = input('Type your word: ')
# pol = 0
# len_a = len(a)//2
# for i in range (0, len_a+1):
#     if a[i] != a[-i-1]:
#         print("It's not a palindrome!")
#         pol = 1
#         break
# if(not pol):
#     print("Great! It's palindrome")
#
# print('\nTask 2')
# s = input('Type your statement: ')
# arr = s.split()
# lens = [len(arr[i]) for i in range (len(arr))]
# ml = lens.index(max(lens))
# print(f'Your longest word is {arr[ml]}')
#
# print('\nTask 3')
# arr = [int(s) for s in input('Type your list of numbers: ').split()]
# ods = sum(map(lambda w: w%2, arr))
# evs = len(arr) - ods
# print(f'Number of odds: {ods}\nNumber of evens: {evs}')
#
# print('\nTask 4')
# dic = {'pls' : 'please', 'sup' : 'greetings my dear friend'}
# a = input('Type your statement: ').lower().split()
# changed = []
# for i in range (len(a)):
#     if(dic.get(a[i]) == None):
#         changed.append(a[i])
#     else:
#         changed.append(dic.get(a[i]))
# print(' '.join(changed))
#
# print('\nTask 5')
# def fib(a, b, n):
#     if n>0:
#         return fib(b, a+b, n-1)
#     else:
#         return b
#
# n = int(input('Type your n for fibonaci: '))
# print(fib(1, 1, n))
#
# print('\nTask 6')
# f_name = input('Type your file name: ')
# with open(f_name) as f:
#     tekst = f.read()
# arr = tekst.split('\n')
# varr = tekst.split()
# cnt_rows = len(arr)
# print(arr, varr)
# cnt_words = len(varr)
# cnt_letts = sum(map(lambda s: len(s), varr))
# print(f'There is {cnt_rows} rows, {cnt_words} words and {cnt_letts} letters')

print('\nTask 7')

def gen(b):
    s = 1
    while(True):
        s*=b
        yield s

b = int(input('Type your multiplicator: '))
q = 1
for n in gen(b):
    if q > 10:
        break
    q+=1
    print(n)
    print('\n')

gen1 = gen(b)
for _ in range(10):
    print(next(gen1))