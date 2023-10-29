# NSU-2023/PAC

## Lesson 1

### Task 1.0 
Реализовать скрипт [1.0.py](https://github.com/doesitworkornot/pac/blob/main/1.0.py)
1. Сгенерировать случайное трехзначное число. Вычислить сумму его цифр.
2. Сгенерировать случайное число. Вычислить сумму его цифр.
3. Задаётся радиус сферы, найти площадь её поверхности и объём.
4. Задаётся год. Определить, является ли он високосным.
5. Определить все числа из диапазона 1, N, являющиеся простыми.
6. Пользователь делает вклад в размере X рублей сроком на Y лет под 10% годовых (каждый год размер его вклада увеличивается на 10%. Эти деньги прибавляются к сумме вклада, и на них в следующем году тоже будут проценты). Вычислить сумму, которая будет на счету пользователя.
7. Вывести все файлы, находящиеся в папке и её подпапках с помощью их абсолютных имён. Имя папки задаётся абсолютным или относительным именем. (можно использовать os.walk())

### Task 1.1 Пузырьковая сортировка.
Реализовать скрипт [bubble_sort.py](https://github.com/doesitworkornot/pac/blob/main/1.1.py)

Входные параметры скрипта: целое число = длина списка Выходные значения: выводит отсортированный список на экран

Что должен делать скрипт:

1. Распарсить входные параметры скрипта с помощью argparse
2. Создать список случайных значений от 0 до 1 длины N (подаётся в качестве входного значения скрипта, получается в пункте 1)
3. Реализовать пузырьковую сортировку с помощью операторов for и if. Никаких sorted!
4. Вывести значения на экран


## Lesson 2

### Task 2.1
Реализовать скрипт [2.1.py](https://github.com/doesitworkornot/pac/blob/main/2.1.py)
1. Вводится строка. Определить является ли она палиндромом и вывести соответствующее сообщение.
2. В строке, состоящей из слов, разделенных пробелом, найти самое длинное слово.
3. Генерируется список случайных целых чисел. Определить, сколько в нем четных чисел, а сколько нечетных.
4. Дан словарь, состоящий из пар слов. Каждое слово является синонимом к парному ему слову. Все слова в словаре различны. Заменить в строке все слова, входящие в словарь, как ключи, на их синонимы.
5. Напишите функцию fib(n), которая по данному целому неотрицательному n возвращает n-e число Фибоначчи. В этой задаче нельзя использовать циклы — используйте рекурсию.
6. Сосчитайте количество строк, слов и букв в произвольном текстовом файле. (слова разделены пробелом, \n не считается символом)
7. Создайте генератор, выводящий бесконечную геометрическую прогрессию. Параметры прогрессии задаются через аргументы генератора

### Task 2.2 Перемножение матриц
Реализовать скрипт [2.2.py](https://github.com/doesitworkornot/pac/blob/main/2.2.py)

Входные параметры скрипта:

Путь к файлу с матрицами. Например, matrix.txt. Внутри файла заданы две целочисленные матрицы.
Путь к файлу с результатом работы программы
Выходные значения: записывает результирующую матрицу в файл с результатом работы программы

Что должен делать скрипт:

1. Прочитать файл, используя встроенные функции Python для работы с файлами.
2. Преобразовать прочитанные строки в матрицы. Матрицы реализовать используя стандартные типы данных Python. Например, список списков.
3. Найти произведение полученных матриц.
4. Записать результат произведения в выходной файл.

### Task 2.3 Операция свёртки
Реализовать скрипт [matrix_mult.py](https://github.com/doesitworkornot/pac/blob/main/2.3.py)

Входные параметры скрипта:

1. Путь к файлу с матрицами. Например, matrix.txt. Внутри файла заданы две целочисленные матрицы (размер второй матрицы не больше первой).
2. Путь к файлу с результатом работы программы
Выходные значения: записывает результирующую матрицу в файл с результатом работы программы

Что должен делать скрипт:

1. Прочитать файл, используя встроенные функции Python для работы с файлами.
2. Преобразовать прочитанные строки в матрицы. Матрицы реализовать используя стандартные типы данных Python. Например, список списков.
3. Найти свёртку первой матрицы со второй. Вторую матрицу использовать как ядро свёртки. Не нужно добавлять падинг, не нужно использовать шаг ядра свёртки != 1. Wiki YouTube
4. Записать результат произведения в выходной файл.


## Lesson 3

### Task 3.1
Реализовать скрипт [3.1.py](https://github.com/doesitworkornot/pac/blob/main/3.1.py)
1. Перенести все операции по работе с количеством объектов в класс Item
2. Дополнить остальными опрерациями сравнения (>, <=, >=, ==), вычитания, а также выполнение операций +=, *=, -=. Все изменения количества должны быть в переделах [0, max_count]
3. Создать ещё 2 класса съедобных фруктов и 2 класса съедобных не фруктов
4. Создать класс Inventory, который содержит в себе список фиксированной длины. Заполнить его None. Доступ в ячейку осуществляется по индексу.
5. Добавить возможность добавлять в него съедобные объекты в определённые ячейки.
6. Добавить возможность уменьшать количество объектов в списке.
7. При достижении нуля, объект удаляется из инвентаря.

### Task 3.2 ООП.
1. Реализовать два класса [Pupa и Lupa](https://github.com/doesitworkornot/pac/blob/main/3.2.py). И класс Accountant.
2. Класс Accountant должен уметь одинаково успешно работать и с экземплярами класса Pupa и с экземплярами класса Lupa. У класса Accountant должен быть метод give_salary(worker). Который, получая на вход экземпляр классов Pupa или Lupa, вызывает у них метод take_salary(int). Необходимо придумать как реализовать такое поведение. Метод take_salary инкрементирует внутренний счётчик у каждого экземпляра класса на переданное ему значение.
3. При этом Pupa и Lupa два датасайнтиста и должны работать с матрицами. У них есть метод do_work(filename1, filename2). Pupa считывают из обоих переданных ему файлов по матрице и поэлементно их суммируют. Lupa считывают из обоих переданных ему файлов по матрице и поэлементно их вычитают. Работники обоих типов выводят результат своих трудов на экран.


## Lesson 4

### Task 4.1

Стараться избегать циклы
Реализовать скрипт [4.1.py](https://github.com/doesitworkornot/pac/blob/main/4.1.py)

1. Отсортировать значения массива по частоте встречания.
2. Дана картинка высоты h, ширины w, тип данных np.uint8 (от 0 до 255). Найти количество уникальных цветов.
3. Написать функцию, вычислающую плавающее среднее вектора
4. Дана матрица (n, 3). Вывести те тройки чисел, которые являются длинами сторон треугольника.

### Task 4.2 Выбор случайных элементов массива
Есть два набора данных: реальные и синтетические. Допустим, мы хотим обучить некоторую ML модель на смеси реальных и синтетических данных. При этом синтетические данные должны браться с вероятностью P. Важно сохранять порядок входных чисел. Например: Для массивов: [1,2,3,4,5,7,8,9,10] и [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] и P=0.2
Один из вариантов возвращаемого значения: [1,-2,3,4,-5,6,7,8,9,10]

Массивы реальных и синтетических данных одинаковой длины.

Реализовать скрипт [random_select.py](https://github.com/doesitworkornot/pac/blob/main/4.2.py)
Входные параметры скрипта: пути к двум файлам со списком целых чисел в каждом. Например file_1.txt содержит:
1 2 3 4 5 6 7
а file_2.txt
-1 -2 -3 -4 -5 -6 -7

Также в качестве аргумента командной строки передаётся вероятность P от 0 до 1.
Результат перемешивания массивов вывести на экран.


## Lesson 5

### Task 5.1
Реализовать скрипт [5.1.py](https://github.com/doesitworkornot/pac/blob/main/5.1.py)
1. Создайте DataFrame с 5 столбцами и 10 строками, заполненный случайными числами от 0 до 1. По каждой строке посчитайте среднее чисел, которые больше 0.3.
2. Посчитайте, сколько целых месяцев длилась добыча на каждой скважине в файле wells_info.csv.
3. Заполните пропущенные числовые значения медианой, а остальные самым часто встречаемым значением в файле wells_info_na.csv.

### Task 5.2 Pandas.
Реализовать скрипт [5.2.py](https://github.com/doesitworkornot/pac/blob/main/5.2.py).
Художественный фильм Титаник режиссера Джеймса Кэмерона славен тем, что идёт три часа и фильме кругом вода. В этой связи многие зрители покидают кинозал для посещения уборной.
В некотором недалеком будущем 7D кинотеатр “ДК Академия” запускает показ ремастера фильма в формате 7D GALACTIC HD. И нанимает специалиста по обработке данных и машинному обучению, чтобы он рассчитал нагрузку на туалетные комнаты во время сеанса. Этот специалист вы! В первую очередь вам необходимо отфильтровать и должным образом подготовить данные, которые вам предоставил кинотеатр. За работу!

Данные, которые предоставил кинотеатр находятся в файлах data/cinema_sessions.csv и data/titanic_with_labels

1. Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;
2. Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;
3. Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и нереально большие значения (выбросы). Вместо них заполнить средним.


## Lesson 6

### Task 6.1
1. Для данных из data/lab6 segmentation [объедините](https://github.com/doesitworkornot/pac/blob/main/6.1.py) пары изображение-маска (список файлов получить с помощью библиотеки os название парных изображений совпадают)
2. Выведите по очереди пары с помощью OpenCV эти пары (переключение по нажатию клавиши)
3. Выделите контуры на масках и отрисуйте их на изображениях
4. Воспроизведите любой видеофайл с помощью OpenCV в градациях серого

### Task 6.2 Лабораторная 6.1
[Программа](https://github.com/doesitworkornot/pac/blob/main/6.2.py) должна реализовывать следующий функционал:

1. Покадровое получение видеопотока с камеры. Использовать камеру ноутбука, вебкамеру или записать видео файл с вебкамеры товарища и использовать его.
2. Реализовать обнаружение движения в видеопотоке: попарно сравнивать текущий и предыдущий кадры. (Если вы сможете в более сложный алгоритм, устойчивый к шумам вебкамеры - будет совсем хорошо)
3. По мере проигрывания видео в отдельном окне отрисовывать двухцветную карту с результатом: красное - есть движение, зелёное - нет движения
4. Добавить таймер, по которому включается и выключается обнаружение движения. О текущем режиме программы сообщать текстом с краю изображения: “Красный свет” - движение обнаруживается, “Зелёный свет” - движение не обнаруживается.


## Lesson 7

### Task 7.1
1. Для датасета Nails segmentation создать генератор, который на каждой итерации возвращает пару списков из заданного количества (аргумент функции) изображений и масок к ним (итератор должен перемешивать примеры).
2. Добавить в генератор случайную аугментацию (каждая применяется случайно). После преобразований все изображения должны иметь одинаковый размер. Обратите внимание, что большинство преобразований должны применяться одинаково к изображению и маске 
  *  Поворот на случайный угол
  *  Отражение по вертикали, горизонтали
  *  Вырезание части изображения
  *  Размытие

### Task 7.2
Из дома с привидениями разбежались все призраки и проказничают в саду. Помогите хозяину дома поймать всех призраков.
Ловлю призраков необходимо реализовать через поиск и сопоставление ключевых точек на изображениях. Алгоритм должен состоять из следующих шагов:

* Загрузка изображения, на котором необходимо осуществлять поиск;
* Загрузка изображения(ий) призраков;
* Обнаружить на них ключевые точки и вычислить для них любые понравившиеся вам дескрипторы SIFT, SURF, ORB;
* Сопоставить точки шаблона (призрака) с точками изображения через Brute-Force Matching или FLANN Matching и найти какой области соответстветствует призрак;
* Найти гомографию используя алгоритм RANSAC. Выделить призрака на изображение рамкой.

Ключевые слова для поиска в Google и документации OpenCV: findHomography, RANSAC, SIFT_Create, FlannBasedMatcher.

## Lesson 8

### Task 8.1
1. Проведите извлечение признаков из wells_info_with_prod.csv (хоть один из столбцов с датой и категориальным признаком должен остаться). Целевой переменной будет Prod1Year
2. Разбейте данные на train и test
3. Отмасштабируйте train (в том числе целевую переменную)
4. Используя модель масштабирования train отмасштабируйте test (использовать метод transform у той же модели масштабирования)