import sys, os, pprint
import numpy as np
from time import time
from IPython import embed

M1 = 10
M2 = 10
N = 500  # num of records
K = 20  # size of dictionary
p = 140  # length of record
q = 40  # length of dictionary
lmda = 5
c = 1
np.random.seed(0)


def normalize(D):
    for i in range(D.shape[1]):
        D[:, i] = D[:, i] / sum(D[:, i]) * c
    return D


def coherence(D):
    """

    :param D: Must be normalized at c
    :return:
    """
    D = D / c
    inners = np.zeros((D.shape[1], D.shape[1]))
    for i in range(D.shape[1]):
        for j in range(i):
            inners[i, j] = abs(np.dot(D[:, i], D[:, j]))
    result = np.max(inners)
    return result


class SIDLProblem:

    def __init__(self, x=None):
        self.X = x
        self.D = None
        self.alpha = None
        self.t = None
        self.random_initialize()

    def random_initialize(self):
        self.D = np.random.rand(q, K)
        self.D = normalize(self.D)
        self.alpha = np.random.rand(N, K)
        self.t = np.zeros((N, K))

    def T(self, i, j):
        """

        :param i: index of record
        :param j: index of base
        :return:
        """
        result = np.zeros(p)
        start = int(self.t[i, j])
        if start >= p:
            embed()
            sys.exit(1)
        elif start+q < p:
            result[start:start+q] = self.D[:, j]
        else:
            try:
                result[start:] = self.D[:(p-start), j]
            except ValueError:
                embed()
                sys.exit(1)
        return result

    def shift_invariant_sparse_coding(self):
        """

        :param X: shape == [records, samples==p]
        :return:
        """
        assert self.X.shape[0] == N
        for m1 in range(M1):
            for i in range(N):
                alpha_Ts = np.zeros((K, p))
                for k in range(K):
                    alpha_Ts[k, :] = self.alpha[i, k] * self.T(i, k)
                alpha_T_sum = np.sum(alpha_Ts, axis=0)
                for k in range(K):
                    x_hat = self.X[i] + alpha_Ts[k, :] - alpha_T_sum
                    t_candidates = []
                    for pp in range(p-q+1):
                        t_candidates.append(np.dot(x_hat[pp:pp+q], self.D[:, k]))
                    self.t[i, k] = np.argmax(np.abs(t_candidates))
                    max_pos = int(np.argmax(np.abs(t_candidates)))
                    if np.abs(t_candidates[max_pos]) > lmda:
                        self.alpha[i, k] = t_candidates[max_pos]
                    else:
                        self.alpha[i, k] = 0

    def shift_invariant_dictionary_update(self):
        assert self.X.shape[0] == N
        for m2 in range(M2):
            x_hats_with_k = np.zeros((N, p))
            for i in range(N):
                x_hats_with_k[i, :] = self.X[i, :]
                for j in range(K):
                    x_hats_with_k[i, :] = x_hats_with_k[i, :] - self.alpha[i, j] * self.T(i, j)
            for k in range(K):
                x_wave = np.zeros(q)
                for i in range(N):
                    alpha = self.alpha[i, k]
                    pos = int(self.t[i, k])
                    x_wave = x_wave + alpha * (x_hats_with_k[i, pos:pos+q] + alpha * self.T(i, k)[pos:pos+q])
                x_wave_norm = np.sqrt(np.sum(np.square(x_wave)))
                alpha_k_square = np.sum(np.square(self.alpha[:, k]))
                if alpha_k_square < 0.1:
                    pass
                elif x_wave_norm / alpha_k_square >= np.sqrt(c):
                    self.D[:, k] = np.sqrt(c) * x_wave / x_wave_norm
                else:
                    self.D[:, k] = x_wave / alpha_k_square
        self.D = normalize(self.D)

    def loss(self):
        l = lmda * np.sum(np.abs(self.alpha))
        for i in range(N):
            restoration = np.zeros(p)
            for j in range(K):
                restoration = restoration + self.alpha[i, j] * self.T(i, j)
            l_sub = 0.5 * np.sum(np.square(self.X[i] - restoration))
            l += l_sub
        return l

    def fit(self, X, max_iterations=100, save=True):
        self.X = X
        l = self.loss()
        start = time()
        print("Iter 0, Time = {0}, Loss = {1}".format(time()-start, l))
        for iter in range(max_iterations):
            self.shift_invariant_sparse_coding()
            print("Iter {0}, Sparse Coding Time = {1}".format(iter + 1, time() - start))
            self.shift_invariant_dictionary_update()
            l = self.loss()
            print("Iter {0}, Time = {1}, Loss = {2}".format(iter+1, time() - start, l))
            start = time()
        if save:
            np.save('D', self.D)
            np.save('alpha', self.alpha)
            np.save('t', self.t)
        else:
            embed()


