import numpy as np
from MatrixGeneration import MatrixGeneration as mg
from typing import Optional


class MatrixCheck(object):
    @staticmethod
    def is_square_matrix(A: np.ndarray) -> bool:
        return True if (A.ndim == 2 and A.shape[0] == A.shape[1]) else False

    @staticmethod
    def is_constant_matrix(self, C: np.ndarray) -> Optional[bool]:
        if C.ndim == 2:
            c = C[0, 0]
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    if C[i, j] != c:
                        return False
            return True
        else:
            return None

    @staticmethod
    def is_binary_matrix(B: np.ndarray):
        return True if (B.ndim == 2 and set(B.flatten()) == {0, 1}) else False

    def is_symmetric_matrix(self, S: np.ndarray) -> bool:
        return True if (self.is_square_matrix(S) and S.T == S) else False

    def is_skewsymmetric_matrix(self, S: np.ndarray) -> bool:
        return True if (self.is_square_matrix(S) and S.T == -S) else False

    def is_bisymmetric_matrix(self, B: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(B):
            E = mg.exchange_matrix(B.shape[0])
            if B == B.T and np.dot(B, E) == np.dot(E, B):
                return True
            else:
                return False
        else:
            return None

    @staticmethod
    def is_toeplitz_matrix(T: np.ndarray) -> Optional[bool]:
        if T.ndim == 2:
            for r in range(1, T.shape[0] - 1):
                for c in range(T.shape[1] - 1):
                    if T[r, c] != T[r + 1, c + 1]:
                        return False
            return True
        else:
            return None

    @staticmethod
    def is_positive_matrix(P: np.ndarray) -> Optional[bool]:
        if P.ndim == 2:
            for r in range(1, P.shape[0]):
                for c in range(P.shape[1]):
                    if P[r, c] <= 0:
                        return False
            return True
        else:
            return None

    @staticmethod
    def is_nonnegative_matrix(P: np.ndarray) -> Optional[bool]:
        if P.ndim == 2:
            for r in range(1, P.shape[0]):
                for c in range(P.shape[1]):
                    if P[r, c] < 0:
                        return False
            return True
        else:
            return None

    @staticmethod
    def is_row_stochastic_matrix(self, S: np.ndarray) -> Optional[bool]:
        if self.is_nonnegative_matrix(S):
            r, c = S.shape
            if (S.sum(axis=0) == np.ones(r)).all():
                return True
            else:
                return False
        else:
            return None

    @staticmethod
    def is_column_stochastic_matrix(self, S: np.ndarray) -> Optional[bool]:
        if self.is_nonnegative_matrix(S):
            r, c = S.shape
            if (S.sum(axis=1) == np.ones(c)).all():
                return True
            else:
                return False
        else:
            return None

    @staticmethod
    def is_doubly_stochastic_matrix(self, S: np.ndarray) -> bool:
        return True if (self.is_row_stochastic_matrix(S) and self.is_column_stochastic_matrix(S)) else False

    def is_involutory_matrix(self, Id: np.ndarray) -> bool:
        return True if (self.is_square_matrix(Id) and np.dot(Id, Id) == np.eye(Id.shape[0])) else False

    def is_skew_involutory_matrix(self, Id: np.ndarray) -> bool:
        return True if (self.is_square_matrix(Id) and np.dot(Id, Id) == -np.eye(Id.shape[0])) else False

    def is_idempotent_matrix(self, Id: np.ndarray) -> bool:
        return True if (self.is_square_matrix(Id) and np.dot(Id, Id) == Id) else False

    def is_skewidempotent_matrix(self, Id: np.ndarray) -> bool:
        return True if (self.is_square_matrix(Id) and np.dot(Id, Id) == -Id) else False

    def is_tripotent_matrix(self, Id: np.ndarray) -> bool:
        return True if (self.is_square_matrix(Id) and np.dot(Id, np.dot(Id, Id)) == Id) else False

    def is_nillpotent_matrix(self, N: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(N):
            if np.linalg.eigvals(N).min() == 0 and np.linalg.eigvals(N).max() == 0:
                return True
            else:
                return False
        else:
            return None

    def is_unipotent_matrix(self, Id: np.ndarray) -> Optional[bool]:
        return self.is_nillpotent_matrix(Id - np.eye(Id.shape[0])) if self.is_square_matrix(Id) else None

    def is_diagonal_matrix(self, D: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(D):
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if i != j and D[i, j] != 0:
                        return False
            return True
        else:
            return None

    def is_orthogonal_matrix(self, R: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(R):
            if np.dot(R.T, R) == np.eye(R.shape[0]):
                return True
            else:
                return False
        else:
            return None

    def is_rotation_matrix(self, R: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(R):
            if self.is_orthogonal_matrix(R) and np.linalg.det(R) == 1:
                return True
            else:
                return False
        else:
            return None

    def is_hankel_matrix(self, H: np.ndarray) -> Optional[bool]:
        if self.is_symmetric_matrix(H):
            for c in range(H.shape[1]):
                for r in range(c):
                    for k in range(c - r):
                        if H[r, c] != H[r + k, c - k]:
                            return False
            return True
        else:
            return None

    def is_diagonallydominant_matrix(self, D: np.ndarray) -> Optional[bool]:
        if self.is_square_matrix(D):
            d = np.diag(np.abs(D))
            s = np.sum(np.abs(D), axis=1) - d
            if (d < s).any():
                return False

            return True
        else:
            return None

    def is_permutation_matrix(self, P: np.ndarray) -> bool:
        if self.is_square_matrix(P):
            if (P.sum(axis=0) == 1).all() and (P.sum(axis=1) == 1).all() and set(P.flatten()) == {0, 1}:
                return True
        return False

    def is_positive_definite_matrix(self, P: np.ndarray) -> Optional[bool]:
        return np.all(np.linalg.eigvals(P) > 0) if self.is_symmetric_matrix(P) else None

    def is_negative_definite_matrix(self, P: np.ndarray) -> Optional[bool]:
        return self.is_positive_definite_matrix(-P)
