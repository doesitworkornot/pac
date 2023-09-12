


def to_int(a):
    """Replacing chars with ints"""
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j] = int(a[i][j])


def matrix_check(a):
    """Trying to find not digits and short rows"""
    check_l = len(a[0])
    for i in range(len(a)):
        if(len(a[i])) != check_l:
            return 0
        for j in range(len(a[i])):
            if not a[i][j].isdigit():
                return 0
    return 1


def mx_mult(a, b):
    """Matrices multiplication"""
    c = [[0 for row in range(len(b[0]))] for col in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                c[i][j] += a[i][k] * b[k][j]
    return c


def main():
    f_name = input('Type file name: ')
    with open(f_name) as f:
        tekst = f.read()
    a, b = tekst.split('\n\n')
    a = a.split('\n')
    b = b.split('\n')
    a = [a[i].split() for i in range(len(a))]
    b = [b[i].split() for i in range(len(b))]
    if matrix_check(a) and matrix_check(b):
        to_int(a)
        to_int(b)
        if len(a[0]) == len(b):
            print(f'Was able to receive such a beautiful matrices\n')
            print(f'Matrix A is {a}')
            print(f'Matrix B is {b}\n\nAnd result is')

#            print(np.dot(a, b))
            print(mx_mult(a, b))

        else:
            print('Matrix cant be multiplied')
    else:
        print('Bad inputs or a wrong file')


if __name__ == "__main__":
    main()
