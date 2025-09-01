import numpy as np
import pandas as pd
from scipy.special import logit, expit
from multi import run

class Packer:
    def __init__(self,
                 regions,
                 seasons):

        self.regions = regions
        self.n_regions = len(self.regions)
        self.region_dict = dict(zip(range(self.n_regions), regions)) 
        self.seasons = seasons
        self.n_seasons = len(self.seasons)

        # precompute upper-triangular indices (excluding diagonal)
        self.iu = np.triu_indices(self.n_regions, k=1)

        # compute parameter count directly
        self._param_count = 1 + len(self.iu[0]) + self.n_regions + 1 + 3 * self.n_regions * self.n_seasons

    # ---- transforms ----
    def _log(self, x):
        return np.log(x)

    def _inv_log(self, y):
        return np.exp(y)

    def _logit(self, x):
        return logit(x)

    def _inv_logit(self, y):
        return expit(y)

    # ---- pack ----
    def pack(self, params):
        parts = []

        # beta0: single scalar
        parts.append([self._log(params["beta0"])])

        # c_vec: upper triangular (excluding diagonal) as vector
        parts.append(self._logit(params["c_vec"]))

        # omega: vector size n_reg
        parts.append(self._logit(params["omega"]))

        # eps: scalar
        parts.append([self._log(params["eps"])])

        # S_init, I_init, E_init: each shape (n_seas, n_reg)
        for key in ["S_init", "I_init", "E_init"]:
            parts.append(self._log(params[key].ravel()))

        flat = np.concatenate(parts)
        return flat

    # ---- unpack ----
    def unpack(self, flat):
        out = {}
        idx = 0

        # beta0
        out["beta0"] = self._inv_log(flat[idx])
        idx += 1

        # c_vec
        c_size = len(self.iu[0])
        c_vec = self._inv_logit(flat[idx:idx+c_size])
        out["c_vec"] = c_vec
        idx += c_size

        # omega
        omega = self._inv_logit(flat[idx:idx+self.n_regions])
        out["omega"] = omega
        idx += self.n_regions

        # eps
        out["eps"] = self._inv_log(flat[idx])
        idx += 1

        # S_init, I_init, E_init
        for key in ["S_init", "I_init", "E_init"]:
            size = self.n_regions * self.n_seasons
            arr = self._inv_log(flat[idx:idx+size]).reshape(self.n_seasons, self.n_regions)
            out[key] = arr
            idx += size

        return out

    def c_vec_to_mat(self, c_vec):
        """Convert a compact c_vec representation into a full symmetric matrix with unit diagonal."""
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def random_packed(self, seed=None):
        """Generate a random packed vector in the transformed parameter space."""
        rng = np.random.default_rng(seed)
        return rng.normal(size=self._param_count)

    @property
    def n_params(self):
        return self._param_count

    @property
    def c_vec_length(self):
        return len(self.iu[0])



# ---- Tests ----
def test_cmat_symmetry_and_unit_diag():
    n_regions, n_seasons = 4, 2
    packer = ParamPacker(n_regions, n_seasons)
    c_vec = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    params = {
        "beta0": 1.5,
        "c_vec": c_vec,
        "omega": np.array([0.1, 0.5, 0.9, 0.3]),
        "eps": 2.0,
        "S_init": np.arange(n_seasons*n_regions).reshape(n_seasons, n_regions),
        "I_init": np.arange(n_seasons*n_regions).reshape(n_seasons, n_regions) + 10,
        "E_init": np.arange(n_seasons*n_regions).reshape(n_seasons, n_regions) + 20,
    }
    flat = packer.pack(params)
    out = packer.unpack(flat)
    c_mat = packer.c_vec_to_mat(out["c_vec"])

    # Check symmetry and unit diagonal
    assert np.allclose(c_mat, c_mat.T)
    assert np.allclose(np.diag(c_mat), np.ones(n_regions))


def test_log_and_inverse_consistency():
    n_regions, n_seasons = 3, 2
    packer = ParamPacker(n_regions, n_seasons)

    arr = np.arange(n_seasons*n_regions).reshape(n_seasons, n_regions) + 1.0
    flat = packer._log(arr.ravel())
    arr_back = packer._inv_log(flat).reshape(n_seasons, n_regions)

    assert np.allclose(arr, arr_back)

def test_random_packed_length():
    n_regions, n_seasons = 3, 2
    packer = ParamPacker(n_regions, n_seasons)
    rnd = packer.random_packed(seed=42)
    assert rnd.shape[0] == packer.n_params


if __name__ == "__main__":
    test_cmat_symmetry_and_unit_diag()
    test_log_and_inverse_consistency()
    test_random_packed_length()
    print("All tests passed.")
