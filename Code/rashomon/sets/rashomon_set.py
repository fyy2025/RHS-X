import numpy as np

from ..loss import compute_Q, compute_Q_bayes, compute_Q_slopes
from ..counter import num_pools


class RashomonSet:
    """
    Caching object to keep track of poolings in Rashomon set
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.P_hash = set()
        self.P_qe = []
        self.Q = np.array([])
        self.H = np.array([])

    def insert(self, sigma: np.ndarray) -> None:
        if self.seen(sigma):
            return
        sigma_hash = self.__process__(sigma)
        self.P_hash.add(sigma_hash)
        self.P_qe.append(sigma)

    def seen(self, sigma: np.ndarray) -> bool:
        self.__verify_shape__(sigma)
        sigma_hash = self.__process__(sigma)
        return sigma_hash in self.P_hash

    def __calculate_loss_stepwise(self, D, y, policies, policy_means, reg, normalize):
        Q_list = []
        for sigma in self.P_qe:
            Q_sigma = compute_Q(D, y, sigma, policies, policy_means, reg, normalize)
            Q_list.append(Q_sigma)
        return Q_list

    def __calculate_loss_slopes(self, D, X, y, policies, reg, normalize):
        Q_list = []
        for sigma in self.P_qe:
            Q_sigma = compute_Q_slopes(D, X, y, sigma, policies, reg, normalize)
            Q_list.append(Q_sigma)
        return Q_list

    def calculate_loss_bayes(self, D, y, policies, policy_means, reg, lattice_edges=None):
        Q_list = []
        for sigma in self.P_qe:
            Q_sigma = compute_Q_bayes(D, y, sigma, policies, policy_means, reg, lattice_edges)
            Q_list.append(Q_sigma)
        self.Q = np.array(Q_list)

    def calculate_loss(self, D, y, policies, policy_means, reg, normalize=0, slopes=False, X=None):
        if not slopes:
            Q_list = self.__calculate_loss_stepwise(D, y, policies, policy_means, reg, normalize)
        else:
            Q_list = self.__calculate_loss_slopes(D, X, y, policies, reg, normalize)
        self.Q = np.array(Q_list)

    def sort(self):
        if len(self.Q) != len(self.P_qe):
            raise RuntimeError("Call RashomonSet.calculate_loss before RashomonSet.sort")
        sorted_idx = np.argsort(self.Q)
        self.Q = self.Q[sorted_idx]
        self.P_qe = [self.P_qe[i] for i in sorted_idx]

    @property
    def size(self):
        return len(self.P_qe)

    @property
    def sigma(self):
        return self.P_qe

    @property
    def loss(self):
        if len(self.Q) != len(self.P_qe):
            raise RuntimeError("Call RashomonSet.calculate_loss before accessing RashomonSet.loss")
        return self.Q

    @property
    def pools(self):
        if len(self.H) != len(self.P_qe):
            self.H = []
            for sigma_i in self.P_qe:
                if sigma_i is None:
                    self.H.append(1)
                else:
                    self.H.append(num_pools(sigma_i))
            self.H = np.array(self.H)
        return self.H

    def __process__(self, sigma: np.ndarray):
        byte_rep = sigma.tobytes()
        return hash(byte_rep)

    def __verify_shape__(self, sigma: np.ndarray):
        # Distinct arrays of different shapes may have same byte representation
        if self.shape != sigma.shape:
            raise RuntimeError(
                f"Expected array of dimensions {self.shape}. Received {sigma.shape}"
            )

    def __iter__(self):
        return iter(self.P_qe)

    def __repr__(self):
        return repr(self.P_qe)

    def __len__(self):
        return len(self.P_qe)
