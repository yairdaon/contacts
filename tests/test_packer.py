import numpy as np
import pytest

from src.packer import Packer

@pytest.mark.parametrize("transform", [True, False])
def test_unpack(transform):
    """Test that unpack(pack(x)) == x for random vectors."""
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    for i in range(100):  # Fewer iterations for debugging
        vector = packer.random_vector()
        dic = packer.unpack(vector)
        packed = packer.pack(dic)
        # FIX: Actually assert the test!
        assert np.allclose(vector, packed, atol=1e-12), f"Pack/unpack failed at iteration {i}"

@pytest.mark.parametrize("transform", [True, False])
def test_pack(transform):
    """Test that pack(unpack(x)) gives back original dict."""
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    for i in range(100):  # Fewer iterations for debugging
        dic = packer.random_dict()
        vector = packer.pack(dic)
        unpacked = packer.unpack(vector)

        for key, value in dic.items():
            corresponding = unpacked[key]
            assert np.allclose(value, corresponding, atol=1e-12), f"Pack/unpack failed for {key} at iteration {i}"


@pytest.mark.parametrize("transform", [True, False])
def test_symmetry_and_diagonal(transform):
    regions = ["HHS1", "HHS3", "HHS5"]
    seasons = ["1900-01-01", "1990-01-02"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    for _ in range(100):  # Reduced iterations
        params = packer.random_dict()
        if transform:
            params = packer.real2pop(params.copy())
        packer.verify_params(params)
        c_mat = packer.c_vec_to_mat(params["c_vec"])

        # Check symmetry and unit diagonal
        assert np.allclose(c_mat, c_mat.T), f"Contact matrix not symmetric: max asymmetry = {np.max(np.abs(c_mat - c_mat.T))}"
        assert np.allclose(np.diag(c_mat), np.ones(packer.n_regions)), f"Contact matrix diagonal not unity: diag = {np.diag(c_mat)}"

@pytest.mark.parametrize("transform", [True, False])
def test_random_vector(transform):
    regions = ["HHS2", "HHS4", "HHS6"]
    seasons = ["1999-01-01", "2000-01-01"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    packed = packer.random_vector()
    assert packed.shape[0] == packer.n_params, f"Packed vector length {packed.shape[0]} != expected {packer.n_params}"

@pytest.mark.parametrize("transform", [True, False])
def test_random_dict(transform):
    """Test that random_dict generates valid parameters."""
    regions = ["HHS2", "HHS4", "HHS6"]
    seasons = ["1999-01-01", "2000-01-01"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    for i in range(100):  # Test with fewer iterations
        unpacked = packer.random_dict()
        if transform:
            unpacked = packer.real2pop(unpacked)
        packer.verify_params(unpacked)

@pytest.mark.parametrize("transform", [True, False])
def test_bounds(transform):
    """Test that bounds are correctly generated."""
    regions = ["HHS1", "HHS2"]
    seasons = ["2020-01-01", "2021-01-01"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    bounds = packer.bounds

    if transform:
        assert bounds is None
    else:
        # Check bounds have correct length
        assert len(bounds.lb) == packer.n_params, f"Lower bounds length {len(bounds.lb)} != expected {packer.n_params}"
        assert len(bounds.ub) == packer.n_params, f"Upper bounds length {len(bounds.ub)} != expected {packer.n_params}"

        # Check that all lower bounds are finite and non-negative (except omega which can be -inf)
        finite_lb = bounds.lb[np.isfinite(bounds.lb)]
        assert np.all(finite_lb >= 0), f"Some finite lower bounds are negative: {finite_lb[finite_lb < 0]}"

        # Check that all upper bounds are positive where finite
        finite_ub = bounds.ub[np.isfinite(bounds.ub)]
        assert np.all(finite_ub > 0), f"Some finite upper bounds are non-positive: {finite_ub[finite_ub <= 0]}"


@pytest.mark.parametrize("transform", [True, False])
def test_linear_constraint(transform):
    """Test that linear constraint is correctly generated."""
    regions = ["HHS1", "HHS2"]
    seasons = ["2020-01-01", "2021-01-01"]
    packer = Packer(transform=transform, regions=regions, seasons=seasons)
    constraint = packer.constraints

    if transform:
        assert constraint == ()
    else:
        # Check constraint matrix dimensions
        expected_rows = packer.n_seasons * packer.n_regions
        expected_cols = packer.n_params
        assert constraint.A.shape == (expected_rows,
                                      expected_cols), f"Constraint matrix shape {constraint.A.shape} != expected {(expected_rows, expected_cols)}"

        # Check that each row has exactly 3 ones (S + E + I)
        row_sums = np.sum(constraint.A, axis=1)
        assert np.allclose(row_sums, 3.0), f"Each constraint should sum 3 variables, got {row_sums}"

        # Check bounds are reasonable
        assert np.all(constraint.lb == 0.0), f"Constraint lower bounds should be 0, got {constraint.lb}"
        assert np.all(constraint.ub > 0), f"Constraint upper bounds should be positive, got {constraint.ub}"
        assert np.all(constraint.ub <= 1.0), f"Constraint upper bounds should be <= 1, got {constraint.ub}"

def test_constraint_satisfaction():
    """Test that random vectors satisfy the constraints."""
    regions = ["HHS1", "HHS2"]
    seasons = ["2020-01-01", "2021-01-01"]
    packer = Packer(transform=False, regions=regions, seasons=seasons)
    constraint = packer.constraints
    bounds = packer.bounds

    for i in range(50):  # Test multiple random vectors
        x = packer.random_vector(seed=i)

        # Check bounds satisfaction
        bounds_satisfied = np.all((x >= bounds.lb) & (x <= bounds.ub))
        assert bounds_satisfied, f"Random vector {i} violates bounds"

        # Check constraint satisfaction
        constraint_values = constraint.A @ x
        constraints_satisfied = np.all((constraint_values >= constraint.lb) & (constraint_values <= constraint.ub))

        if not constraints_satisfied:
            violations = (constraint_values < constraint.lb) | (constraint_values > constraint.ub)
            print(f"Vector {i} constraint violations at indices: {np.where(violations)[0]}")
            print(f"Violation values: {constraint_values[violations]}")
            print(f"Constraint bounds: lb={constraint.lb[violations]}, ub={constraint.ub[violations]}")

        assert constraints_satisfied, f"Random vector {i} violates linear constraints"
