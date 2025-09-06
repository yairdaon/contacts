import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packer import Packer


def test_unpack():
    """Test that unpack(pack(x)) == x for random vectors."""
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for i in range(100):  # Fewer iterations for debugging
        vector = packer.random_vector()
        dic = packer.unpack(vector)
        packed = packer.pack(dic)
        # FIX: Actually assert the test!
        assert np.allclose(vector, packed, atol=1e-12), f"Pack/unpack failed at iteration {i}"


def test_pack():
    """Test that pack(unpack(x)) gives back original dict."""
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for i in range(100):  # Fewer iterations for debugging
        dic = packer.random_dict()
        vector = packer.pack(dic)
        unpacked = packer.unpack(vector)

        for key, value in dic.items():
            corresponding = unpacked[key]
            assert np.allclose(value, corresponding, atol=1e-12), f"Pack/unpack failed for {key} at iteration {i}"


def test_symmetry_and_diagonal():
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(regions, seasons)
    for _ in range(100):  # Reduced iterations
        params = packer.random_dict()
        packer.verify(packer.real2pop(params.copy()))
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
    """Test that random_dict generates valid parameters."""
    regions = ["HHS2", "HHS4", "HHS6"]
    seasons = ["1999-01-01", "2000-01-01"]
    packer = Packer(regions=regions, seasons=seasons)
    for i in range(100):  # Test with fewer iterations
        unpacked = packer.random_dict()
        unpacked = packer.real2pop(unpacked)
        try:
            packer.verify(unpacked)
        except AssertionError as e:
            print(f"Verification failed at iteration {i}: {e}")
            print(f"Problem values: {unpacked}")
            raise


if __name__ == "__main__":
    print("Running packer tests...")
    try:
        test_unpack()
        print("✓ test_unpack passed")
        test_pack()
        print("✓ test_pack passed")
        test_symmetry_and_diagonal()
        print("✓ test_symmetry_and_diagonal passed")
        test_random_vector()
        print("✓ test_random_vector passed")
        test_random_dict()
        print("✓ test_random_dict passed")
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        raise