class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16

    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    @property
    def count(self):
        return self._count

    # Ещё один способ изменить атрибут класса
    @count.setter
    def count(self, val):
        if val <= self._max_count:
            self._count = val
        else:
            pass

    def __add__(self, num):
        """ Сложение с числом """
        return self._count + num

    def __mul__(self, num):
        """ Умножение на число """
        return self._count * num

    def __lt__(self, num):
        """ Сравнение меньше """
        return self._count < num

    def __gt__(self, num):
        return self._count > num

    def __ge__(self, num):
        return self._count >= num

    def __le__(self, num):
        return self._count <= num

    def __eq__(self, num):
        return self._count == num

    def __len__(self):
        """ Получение длины объекта """
        return self._count

    def __iadd__(self, other):
        a = self._count + other
        if self._max_count >= a >= 0:
            self.count = a
            return self
        else:
            print('Out of range')
            return self

    def __imul__(self, other):
        a = self._count * other
        if self._max_count >= a >= 0:
            self.count = a
            return self
        else:
            print('Out of range')
            return self

    def __isub__(self, other):
        a = self._count - other
        if self._max_count >= a >= 0:
            self.count = a
            return self
        else:
            print('Out of range')
            return self



class Eatable:
    def __init__(self):
        print('I\'m eatable as can be')

class NonEatable:
    def __init__(self):
        print('I\'m not eatable as can be')


class Fruit1(Eatable):
    def __init__(self):
        super().__init__()
        self.count = 1
        print('I\'m a fruit 1')


class Fruit2(Eatable):
    def __init__(self):
        super().__init__()
        self.count = 1
        print('I\'m a fruit 2')


class NonFruit1(Eatable):
    def __init__(self):
        super().__init__()
        self.count = 1
        print('I\'m not a fruit 1')


class NonFruit2(Eatable):
    def __init__(self):
        super().__init__()
        self.count = 1
        print('I\'m not a fruit 2')


class Inventory:
    def __init__(self, len=10):
        self.len = len
        self.list = [None for i in range(len)]

    def add(self, val, id):
        if isinstance(val, Eatable) and 0 <= id < self.len:
            if self.list[id] is None:
                self.list[id] = val
            elif type(val) == type(self.list[id]):
                self.list[id].count += 1
                print(f'Added another to {type(val)} basket')
            else:
                print(f'Replaced {type(self.list[id])} with {type(val)}')
                self.list[id] = val
        else:
            print('It\'s not eatable or bad index')

    def reduce(self, in_pl):
        if 0<= in_pl < self.len:
            if self.list[in_pl] is None:
                print('Item wasn\'t existed')
            elif self.list[in_pl].count == 1:
                self.list[in_pl] = None
                print('Removed Item')
            else:
                print(f'A little bit less of {type(self.list[in_pl])}')
                self.list[in_pl].count -= 1
        else:
            print('Bad index')

    def __del__(self):
        print('Deleted')


inv = Inventory()
print(inv.len)
apple = Fruit2()
inv.add(apple, 0)
print(inv.list[0])
inv.add(apple, 100)
not_eat = NonEatable()
inv.add(not_eat, 2)
inv.add(apple, 0)
inv.reduce(0)
inv.reduce(0)
inv.reduce(0)
# item1 = Item()
# item1 -= 1000
# print(item1.count)


