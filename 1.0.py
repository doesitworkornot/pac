import random
import math
import os

print('1st task')
a = random.randint(100, 999)
print(f'get {a}')
print(f'summ is {sum(map(int,str(a)))}')

print('\n2st task')
a = random.randint(0, 100000)
print(f'get {a}')
print(f'summ is {sum(map(int,str(a)))}')

print('\n3st task')
r = int(input('type radius '))
print(f'area is {r**2*math.pi*4} and volume is {r**3*math.pi*(4/3)}')

print('\n4th3 task')
year = int(input('waiting for a year'))
if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
    print('it is')
else:
    print('its not')

print('\n5th task')
n = int(input('get n'))
for x in range(2, n + 1):
    if all(x % i != 0 for i in range(2, int(math.sqrt(x)) + 1)):
        print(x, end=" ")

print('\n6th task')
percent = 0.1
money = int(input('how much'))
year = int(input('how long'))
for i in range (1, year+1):
    money += percent*money
print(f'You\'l get {money}')
# По-моему, можно было возвести percent + 1 в степень и домножить

print('\n7th task')
directory = input('list directory')
filelist = []
for root, dirs, files in os.walk(directory):
    for file in files:
        print(os.path.join(root, file))