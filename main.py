#! /usr/bin python
import numpy as np
from MatrixCheck import MatrixCheck
from MatrixGeneration import MatrixGeneration

if __name__ == '__main__':
    # Matrix check
    mc = MatrixCheck()
    mg = MatrixGeneration()
    print('Diagonal matrix from a vector:')
    D = np.diag([1, 3, 5, 7])
    print(D)
    print('Shift matrices:')
    for i in range(3):
        Si = np.eye(3, k=i)
        Ji = np.eye(3, k=-i)
        if mc.is_nillpotent_matrix(Si):
            print('Si is a Nillpotent matrix')
        else:
            print('Si is not a Nillpotent matrix')
        if mc.is_nillpotent_matrix(Ji):
            print('Ji is a Nillpotent matrix')
        else:
            print('Ji is not a Nillpotent matrix')

    B = np.array([[6, 7, 8, 9],
                  [4, 6, 7, 8],
                  [1, 4, 6, 7],
                  [0, 1, 4, 6],
                  [2, 0, 1, 4]])

    if mc.is_toeplitz_matrix(B):
        print('B is a Toeplitz matrix')
    else:
        print('B is not a Toeplitz matrix')

    DD0 = np.array([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
    if mc.is_diagonallydominant_matrix(DD0):
        print('DD0 is a Diagonally Dominant matrix')
    else:
        print('DD0 is not a Diagonally Dominant matrix')
    DD1 = np.array([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
    if mc.is_diagonallydominant_matrix(DD1):
        print('DD1 is a Diagonally Dominant matrix')
    else:
        print('DD1 is not a Diagonally Dominant matrix')

    print('Random rotation matrix ')
    print(mg.random_rotation_matrix(3))
