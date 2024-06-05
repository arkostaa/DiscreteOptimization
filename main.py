import numpy as np


class Matrix:
    def __init__(self, rows=0, cols=0, val=0.0):
        self.matr = np.full((rows, cols), val, dtype=float)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return self.matr[row, col]
        else:
            return self.matr[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            row, col = index
            self.matr[row, col] = value
        else:
            self.matr[index] = value

    def empty(self):
        return self.matr.size == 0

    def get_rows(self):
        return self.matr.shape[0]

    def get_cols(self):
        return self.matr.shape[1]

    def reset(self, rows=0, cols=0, val=0.0):
        self.matr = np.full((rows, cols), val, dtype=float)

    def fill(self, val):
        self.matr.fill(val)


def print_matrix(matr):
    rows, cols = matr.get_rows(), matr.get_cols()
    for i in range(rows):
        for j in range(cols):
            print(matr[i, j], end='\t')
        print()
    print()


def print_vector(elems):
    for elem in elems:
        print(elem, end=' ')
    print()


def print_result_max_simplex_method(ans):
    size_func = len(ans[1])
    for i in range(size_func):
        print(f'x{i+1}: {ans[1][i]}')
    print(f'max: {ans[0]}')


def is_equal(value_one, value_two, eps=0.01):
    return abs(value_one - value_two) <= eps


def transformation_rows(matr, ind_row_base, ind_col_base):
    rows, cols = matr.get_rows(), matr.get_cols()
    for i in range(rows):
        if i != ind_row_base:
            value = matr[i, ind_col_base]
            if is_equal(value, 0.0):
                continue
            if value > 0.0:
                matr[i] = [elem - value * matr[ind_row_base][j] for j, elem in enumerate(matr[i])]
            else:
                value = abs(value)
                matr[i] = [elem + value * matr[ind_row_base][j] for j, elem in enumerate(matr[i])]


def make_result(matr, basis, size_func):
    rows, cols = matr.get_rows(), matr.get_cols()
    count_lim = len(basis)
    max_value = matr[-1, -1]
    result = [0.0] * size_func
    for i in range(count_lim):
        ind = basis[i]
        if ind < size_func:
            result[ind] = matr[i, -1]
    return max_value, result


def validate_simplex_method_max(func, limits):
    size_func = len(func)
    if not func or not limits:
        return False
    for row in limits:
        if len(row) - 1 != size_func:
            return False
    return True


def create_matrix_simplex_method_max(func, limits):
    count_lim = len(limits)
    size_func = len(func)
    rows = count_lim + 1
    cols = size_func + count_lim + 1
    matr = Matrix(rows, cols)
    for i in range(count_lim):
        for j in range(size_func):
            matr[i, j] = limits[i][j]
    for i in range(count_lim):
        matr[i, size_func + i] = 1.0
    for i in range(size_func):
        matr[-1, i] = -func[i]
    for i in range(rows - 1):
        matr[i, -1] = limits[i][size_func]
    return matr


def create_basis_simplex_method_max(size_func, count_lim):
    return list(range(size_func, size_func + count_lim))


def calculate_simplex_method_max(matr, basis, count_lim):
    rows, cols = matr.get_rows(), matr.get_cols()
    while True:
        print_vector(basis)
        print_matrix(matr)
        ind_col = -1
        ind_row = -1
        for i in range(cols - 2):
            if matr[-1, i] < 0.0:
                ind_col = i
                break
        if ind_col == -1:
            print("Found optimal plan.")
            break
        ratio_exist = False
        near_zero = False
        min_basis_ind_col = -1
        min_ratio = np.inf
        for i in range(count_lim):
            ratio = 0.0
            if not is_equal(matr[i, ind_col], 0.0):
                if is_equal(matr[i, -1], 0.0) and matr[i, ind_col] > 0.0:
                    near_zero = True
                ratio = matr[i, -1] / matr[i, ind_col]
            if ratio > 0.0 or near_zero:
                ratio_exist = True
                near_zero = False
                if is_equal(ratio, min_ratio):
                    curr_basis_ind_col = basis[i]
                    if curr_basis_ind_col < min_basis_ind_col:
                        min_basis_ind_col = curr_basis_ind_col
                        ind_row = i
                if ratio < min_ratio:
                    min_ratio = ratio
                    ind_row = i
                    min_basis_ind_col = basis[i]
        if not ratio_exist:
            print("Not found optimal plan.")
            break
        if not is_equal(matr[ind_row, ind_col], 1.0):
            matr[ind_row] = [elem / matr[ind_row, ind_col] for elem in matr[ind_row]]
        basis[ind_row] = ind_col
        transformation_rows(matr, ind_row, ind_col)


def simplex_method_max(func, limits):
    if not validate_simplex_method_max(func, limits):
        return 0.0, []
    count_lim = len(limits)
    size_func = len(func)
    matr = create_matrix_simplex_method_max(func, limits)
    basis = create_basis_simplex_method_max(size_func, count_lim)
    calculate_simplex_method_max(matr, basis, count_lim)
    return make_result(matr, basis, size_func)


def print_result_fill_backpack(p):
    print(f"max: {p[0]}")
    for i in range(len(p[1])):
        print(f'x{i + 1}: {p[1][i]}')


def fill_backpack(weight, price, capacity):
    if len(weight) != len(price) or len(weight) <= 1 or capacity < 1:
        return (0, [])

    count_elem = len(weight)
    matr = [[0] * capacity for _ in range(count_elem)]

    for i in range(capacity):
        matr[0][i] = price[0] * ((i + 1) // weight[0])

    for i in range(1, count_elem):
        for j in range(capacity):
            tmp = j + 1 - weight[i]
            matr[i][j] = max(matr[i - 1][j],
                             price[i] if tmp == 0 else
                             (matr[i][tmp - 1] + price[i] if tmp > 0 else 0))

    print_matrix(matr)

    max_value = matr[count_elem - 1][capacity - 1]
    ans_x = [0] * count_elem

    i, j = count_elem - 1, capacity - 1
    while i > 0 and j > 0:
        if matr[i - 1][j] != matr[i][j]:
            ans_x[i] += 1
            j -= weight[i]
        else:
            i -= 1
            if i == 0:
                ans_x[i] = (j + 1) // weight[i]

    return (max_value, ans_x)


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def main():
    print("simplex method")
    print("ans one")
    ans_one = simplex_method_max([5, 7], [[2, 3, 8], [4, 6, 10]])
    print_result_max_simplex_method(ans_one)
    print("ans two")
    ans_two = simplex_method_max([7, 5, 6], [[3, 4, 2, 22], [2, 3, 4, 25], [3, 2, 4, 20]])
    print_result_max_simplex_method(ans_two)

    print("backpack")
    print(" ")
    ans_one_backpack = fill_backpack([5, 6, 4, 8, 7], [6, 8, 5, 10, 9], 18)
    print_result_fill_backpack(ans_one_backpack)
    print(" ")
    ans_two_backpack = fill_backpack([4, 5, 7], [3, 4, 6], 18)
    print_result_fill_backpack(ans_two_backpack)


if __name__ == "__main__":
    main()
