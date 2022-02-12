import numpy as np


class MatrixGeneration(object):
    @staticmethod
    def exchange_matrix(n):
        E = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                if c == n - r + 1:
                    E[r, c] = 1
        return E

    @staticmethod
    def skew_symmetric_matrix(x):
        if len(x) == 3:
            S = np.array([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]])
            return S
        else:
            raise ValueError('Dimension {} is not supported. \
                              Use only 3 instead'.format(len(x)))

    def random_rotation_matrix(self, dim=3):
        if dim == 2:
            theta = np.random.random()
            c, s = np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta)
            R = np.array([[c, -s], [s, c]])
            return R
        elif dim == 3:
            u = np.random.uniform(size=(dim,))
            if np.linalg.norm(u) > 0:
                u = u / np.linalg.norm(u)
                S = self.skew_symmetric_matrix(u)
                In = np.eye(dim)
                R = np.dot((In - S), np.linalg.inv(In + S))
                return R
        else:
            raise ValueError('Dimension {} is not supported. \
                              Use 2 or 3 instead'.format(dim))

    @staticmethod
    def constant_matrix(self, size, c):
        m, n = size
        C = np.zeros((m, n))
        C[:, :] = c
        return C
