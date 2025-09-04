import numpy as np
import pandas as pd
from scipy.special import logit, expit
from numpy import exp, log

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
        self.n_params = 1 + len(self.iu[0]) + self.n_regions + 1 + 3 * self.n_regions * self.n_seasons

    def real2pop(self, params):
        S_init = exp(params['S_init'])
        E_init = exp(params['E_init'])
        I_init = exp(params['I_init'])
        tot = S_init + E_init + I_init + 1  # eq R_init = 0
        params["S_init"] = S_init/tot
        params["E_init"] = E_init/tot
        params["I_init"] = I_init/tot
        return params

    def pop2real(self, params):
        params["S_init"] = log(params['S_init'])
        params["E_init"] = log(params['E_init'])
        params["I_init"] = log(params['I_init'])
        return params

    def verify(self, params):
        assert params['beta0'] > 0
        assert np.all(0 < params['c_vec'])
        assert np.all(params['c_vec'] < 1)
        assert 0 < params['eps'] < 1
        assert np.all(0 < params["S_init"])
        assert np.all(0 < params["E_init"])
        assert np.all(0 < params["I_init"])
        assert np.all(params["S_init"] < 1)
        assert np.all(params["E_init"] < 1)
        assert np.all(params["I_init"] < 1)
        tot = params["S_init"] + params["I_init"] + params["E_init"]
        assert np.all(tot < 1)
        assert params["S_init"].shape == (self.n_seasons, self.n_regions)
        assert params["E_init"].shape == (self.n_seasons, self.n_regions)
        assert params["I_init"].shape == (self.n_seasons, self.n_regions)
        for key, value in params.items():
            assert not np.any(np.isnan(value)), key

        
    def pack(self, params):
        parts = []

        # beta0: single scalar
        parts.append([log(params["beta0"])])

        # c_vec: upper triangular (excluding diagonal) as vector
        parts.append(logit(params["c_vec"]))

        # omega: vector size n_reg
        parts.append(params["omega"])

        # eps: scalar
        parts.append([logit(params["eps"])])

        # S_init, I_init, E_init: each shape (n_seas, n_reg)
        for key in ["S_init", "I_init", "E_init"]:
            parts.append(params[key].ravel())

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,)
        return flat

    
    # ---- unpack ----
    def unpack(self, flat):
        assert flat.shape == (self.n_params,)
        out = {}
        idx = 0

        # beta0
        out["beta0"] = exp(flat[idx])
        idx += 1

        # c_vec
        c_size = len(self.iu[0])
        c_vec = expit(flat[idx:idx+c_size])
        out["c_vec"] = c_vec
        idx += c_size

        # omega
        omega = flat[idx:idx+self.n_regions]
        out["omega"] = omega
        idx += self.n_regions

        # eps
        out["eps"] = expit(flat[idx])
        idx += 1

        # S_init, I_init, E_init
        for key in ["S_init", "I_init", "E_init"]:
            size = self.n_regions * self.n_seasons
            arr = flat[idx:idx+size].reshape(self.n_seasons, self.n_regions)
            out[key] = arr
            idx += size

        return out

    def c_vec_to_mat(self, c_vec):
        """Convert a compact c_vec representation into a full symmetric matrix with unit diagonal."""
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def c_mat_to_vec(self, c_mat):
        return c_mat[self.iu]

    def random_vector(self, seed=None):
        """Generate a random packed vector in the transformed parameter space."""
        params = self.random_dict()
        vec = self.pack(params)
        return vec

    def random_dict(self, seed=None):
        np.random.seed(seed)
        out = dict(
            beta0=np.random.uniform(0.2, 0.45),
            c_vec=np.random.uniform(0,1, size=self.c_vec_length),
            omega=np.random.uniform(0,2*np.pi, size=self.n_regions),
            eps=np.random.uniform(0,1)
        )

        for key in ["S_init", "I_init", "E_init"]:
            arr = np.random.randn(self.n_regions * self.n_seasons).reshape(self.n_seasons, self.n_regions)
            out[key] = arr
        out = self.real2pop(out)
        return out


    @property
    def c_vec_length(self):
        return len(self.iu[0])


def test_unpack():
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for _ in range(2000):
        vector = packer.random_vector()
        dic = packer.unpack(vector)
        packed = packer.pack(dic)
        np.array_equal(vector, packed, equal_nan=True)

def test_pack():
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for _ in range(2000):
        dic = packer.random_dict()
        vector = packer.pack(dic)
        unpacked = packer.unpack(vector)
        for key, value in dic.items():
            corresponding = unpacked[key]
            assert np.allclose(value, corresponding, atol=1e-15)


def test_symmetry_and_diagonal():
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for _ in range(2000):
        params = packer.random_dict()
        packer.verify(params)
        c_mat = packer.c_vec_to_mat(params["c_vec"])

        # Check symmetry and unit diagonal
        assert np.allclose(c_mat, c_mat.T)
        assert np.allclose(np.diag(c_mat), np.ones(packer.n_regions))


def test_random_vector():
    regions = ["HHS2", "HHS4", "HHS6"]
    seasons = ["1999-01-01", "2000-01-01"]
    packer = Packer(regions=regions, seasons=seasons)
    packed = packer.random_vector()
    assert packed.shape[0] == packer.n_params

def test_random_dict():
    regions = ["HHS2", "HHS4", "HHS6"]
    seasons = ["1999-01-01", "2000-01-01"]
    packer = Packer(regions=regions, seasons=seasons)
    for _ in range(2000):
        unpacked = packer.random_dict()
        packer.real2pop(unpacked)
        packer.verify(unpacked)




if __name__ == "__main__":
    test_unpack()
    test_pack()
    test_symmetry_and_diagonal()
    test_random_vector()
    test_random_dict()
    print("All tests passed.")
