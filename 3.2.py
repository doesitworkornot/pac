
class Worker:
    money = 0

    def take_money(self, more_money):
        self.money += more_money


class Pupa(Worker):
    def do_work(self, filename1, filename2):
        do_work('1', filename1, filename2)


class Lupa(Worker):
    def do_work(self, filename1, filename2):
        do_work('-1', filename1, filename2)


class Accountant:
    def give_salary(self, worker, how_much):
        if isinstance(worker, Worker):
            worker.take_money(how_much)
            print('Just paid')
        else:
            print('Its not worker')


def get_mx(path):
    """Reading from path matrix"""
    with open(path) as f:
        data = f.read()
    a = data.split('\n')
    a = [a[i].split() for i in range(len(a))]
    to_int(a)
    return a


def to_int(a):
    """Replacing chars with ints"""
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j] = int(a[i][j])


def do_work(op, filename1, filename2):
    a = get_mx(filename1)
    b = get_mx(filename2)
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise Exception('Bad inputs. Matrices should be the same size')
    for i in range(len(a)):
        for j in range(len(a[0])):
            print(f'{a[i][j] + int(op) * b[i][j]}', end=' ')
        print('\n')


def main():
    pupa = Pupa()
    lupa = Lupa()
    acc = Accountant()
    print(f'Pupas money {pupa.money}')
    acc.give_salary(pupa, 100)
    print(f'Pupas money {pupa.money}')
    pupa.do_work("mx.txt", "mx2.txt")
    lupa.do_work("mx.txt", "mx2.txt")
    acc.give_salary(pupa, 10000)
    print(f'Pupas money {pupa.money}')


if __name__ == "__main__":
    main()
