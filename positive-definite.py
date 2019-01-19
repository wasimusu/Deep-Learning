import numpy as np


def is_positive_definite_matrix(matrix):
    """ Determine if the input matrix is Positive definite or not.
     A matrix is PD is all it's eigen values are positive """

    if matrix.shape[0] != matrix.shape[1]:
        print("Not a PD because it is not a square matrix ")
        return

    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    eigen_values = np.round(eigen_values, 2)

    if eigen_values.all() > 0:
        print("It's PD because all of it's eigen values {} are positive".format(eigen_values))
    else:
        print("It's not PD because all of it's eigen values {} are not positive".format(eigen_values))


if __name__ == '__main__':
    B = np.array(np.eye(3), dtype=np.int8)

    A = np.array([[1, -2],
                  [3, 4],
                  [-5, 6]], dtype=np.int8)

    C = np.array([[2, 2, 1],
                  [2, 3, 2],
                  [1, 2, 2]], dtype=np.int8)

    is_positive_definite_matrix(A)
    is_positive_definite_matrix(np.transpose(A).dot(A))
    is_positive_definite_matrix(A.dot(np.transpose(A)))
    is_positive_definite_matrix(B)
    is_positive_definite_matrix(-B)
    is_positive_definite_matrix(C)
    is_positive_definite_matrix(C - 0.1 * B)
    is_positive_definite_matrix(C - 0.01 * A.dot(np.transpose(A)))
