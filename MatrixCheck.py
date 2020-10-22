import numpy as np
from MatrixGeneration import MatrixGeneration


class MatrixCheck:
    @staticmethod
    def is_square_matrix(A) -> bool:
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            return True
        else:
            return False

    @staticmethod
    def is_constant_matrix(self, C) -> bool:
        if C.ndim == 2:
            c = C[0, 0]
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    if C[i, j] != c:
                        return False
            return True

    @staticmethod
    def is_binary_matrix(self, B):
        if B.ndim == 2 and set(B.flatten()) == {0, 1}:
            return True
        else:
            return False

    def is_symmetric_matrix(self, S) -> bool:
        if self.is_square_matrix(S) and S.T == S:
            return True
        else:
            return False

    def is_skewsymmetric_matrix(self, S) -> bool:
        if self.is_square_matrix(S) and S.T == -S:
            return True
        else:
            return False

    def is_bisymmetric_matrix(self, B) -> bool:
        if self.is_square_matrix(B):
            E = mg.exchange_matrix(B.shape[0])
            if B == B.T and np.dot(B, E) == np.dot(E, B):
                return True
            else:
                return False

    @staticmethod
    def is_toeplitz_matrix(T) -> bool:
        if T.ndim == 2:
            for r in range(1, T.shape[0] - 1):
                for c in range(T.shape[1] - 1):
                    if T[r, c] != T[r + 1, c + 1]:
                        return False
            return True

    @staticmethod
    def is_positive_matrix(P) -> bool:
        if P.ndim == 2:
            for r in range(1, P.shape[0]):
                for c in range(P.shape[1]):
                    if P[r, c] <= 0:
                        return False
            return True

    @staticmethod
    def is_nonnegative_matrix(P) -> bool:
        if P.ndim == 2:
            for r in range(1, P.shape[0]):
                for c in range(P.shape[1]):
                    if P[r, c] < 0:
                        return False
            return True

    @staticmethod
    def is_row_stochastic_matrix(self, S) -> bool:
        if self.is_nonnegative_matrix(S):
            r, c = S.shape
            if (S.sum(axis=0) == np.ones(r)).all():
                return True
            else:
                return False

    @staticmethod
    def is_column_stochastic_matrix(self, S):
        if self.is_nonnegative_matrix(S):
            r, c = S.shape
            if (S.sum(axis=1) == np.ones(c)).all():
                return True
            else:
                return False

    @staticmethod
    def is_doubly_stochastic_matrix(self, S) -> bool:
        if self.is_row_stochastic_matrix(S) and self.is_column_stochastic_matrix(S):
            return True
        else:
            return False

    def is_involutory_matrix(self, I) -> bool:
        if self.is_square_matrix(I) and np.dot(I, I) == np.eye(I.shape[0]):
            return True
        else:
            return False

    def is_skew_involutory_matrix(self, I) -> bool:
        if self.is_square_matrix(I) and np.dot(I, I) == -np.eye(I.shape[0]):
            return True
        else:
            return False

    def is_idempotent_matrix(self, I) -> bool:
        if self.is_square_matrix(I) and np.dot(I, I) == I:
            return True
        else:
            return False

    def is_skewidempotent_matrix(self, I) -> bool:
        if self.is_square_matrix(I) and np.dot(I, I) == -I:
            return True
        else:
            return False

    def is_tripotent_matrix(self, I) -> bool:
        if self.is_square_matrix(I) and np.dot(I, np.dot(I, I)) == I:
            return True
        else:
            return False

    def is_nillpotent_matrix(self, N) -> bool:
        if self.is_square_matrix(N):
            if np.linalg.eigvals(N).min() == 0 and np.linalg.eigvals(N).max() == 0:
                return True
            else:
                return False
        else:
            return False

    def is_unipotent_matrix(self, I):
        if self.is_square_matrix(I):
            return self.is_nillpotent_matrix(self, I - np.eye(I.shape[0]))

    def is_diagonal_matrix(self, D) -> bool:
        if self.is_square_matrix(D):
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if i != j and D[i, j] != 0:
                        return False
            return True

    def is_orthogonal_matrix(self, R) -> bool:
        if self.is_square_matrix(R):
            if np.dot(R.T, R) == np.eye(R.shape[0]):
                return True
            else:
                return False

    def is_rotation_matrix(self, R) -> bool:
        if self.is_square_matrix(R):
            if self.is_orthogonal_matrix(self, R) and np.linalg.det(R) == 1:
                return True
            else:
                return False

    def is_hankel_matrix(self, H) -> bool:
        if self.is_symmetric_matrix(H):
            for c in range(H.shape[1]):
                for r in range(c):
                    for k in range(c - r):
                        if H[r, c] != H[r + k, c - k]:
                            return False
            return True

    def is_diagonallydominant_matrix(self, D) -> bool:
        if self.is_square_matrix(D):
            d = np.diag(np.abs(D))
            s = np.sum(np.abs(D), axis=1) - d
            if (d < s).any():
                return False

            return True

    def is_permutation_matrix(self, P) -> bool:
        if self.is_square_matrix(P):
            if (P.sum(axis=0) == 1).all() and (P.sum(axis=1) == 1).all() and set(P.flatten()) == {0, 1}:
                return True
        return False

    def is_positive_definite_matrix(self, P):
        if self.is_symmetric_matrix(self, P):
            return np.all(np.linalg.eigvals(P) > 0)

    def is_negative_definite_matrix(self, P):
        return self.is_positive_definite_matrix(self, -P)


