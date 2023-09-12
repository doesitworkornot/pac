"""
Convolution for two matrices
"""


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


def conv(a, b):
    nx, ny, mx, my = len(a[0]), len(a), len(b[0]), len(b)
    ox, oy = nx-mx+1, ny-my+1
    c = [[0 for row in range(ox)] for col in range(oy)]
    for i in range(ox):
        for j in range(oy):
            for u in range(mx):
                for v in range(my):
                    c[i][j] += a[i+u][j+v]*b[u][v]
    return c


def main():
    f_name = input('Type input file name: ')
    out_name = input('Type output file name: ')
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
        print(f'Was able to receive such a beautiful matrices\n')
        print(f'Matrix A is {a}')
        print(f'Matrix B is {b}\n\nAnd result is')
        c = conv(a, b)
        fi = open(out_name, "w")
        for i in range(len(c)):
            for j in range(len(c[i])):
                print(f'{c[i][j]}', end=' ')
                fi.write(f'{c[i][j]} ')
            print('\n')
            fi.write('\n')

    else:
        print('Bad inputs')
        return 1


if __name__ == "__main__":
    main()
