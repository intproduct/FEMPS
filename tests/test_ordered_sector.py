import torch

from femps.exterior import antisymmetry_residual, normalized_slater_from_minors
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian,
    antisymmetric_many_body_hamiltonian_dense_two_body,
)
from femps.ordered_sector import (
    extend_from_ordered_sector,
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
    restrict_to_ordered_sector,
)


def test_ordered_sector_restriction_is_an_isometry() -> None:
    generator = torch.Generator().manual_seed(29)
    orbitals = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    orbitals, _ = torch.linalg.qr(orbitals)
    state = normalized_slater_from_minors(orbitals)
    ordered = restrict_to_ordered_sector(state)
    reconstructed = extend_from_ordered_sector(ordered, 5, 3)
    assert torch.allclose(torch.vdot(ordered, ordered), torch.vdot(state.ravel(), state.ravel()))
    assert torch.allclose(reconstructed, state)
    assert antisymmetry_residual(reconstructed) == 0


def test_local_ordered_hamiltonian_equals_exterior_truth() -> None:
    _, one_body = finite_difference_harmonic_hamiltonian(9, 0.7)
    ordered = ordered_sector_hamiltonian(one_body, 3)
    exterior = antisymmetric_many_body_hamiltonian(one_body, 3)
    assert torch.allclose(ordered, exterior, atol=1e-13, rtol=1e-13)


def test_ordered_pair_potential_is_diagonal() -> None:
    grid, one_body = finite_difference_harmonic_hamiltonian(7, 0.8)
    potential = 0.2 * (grid[:, None] - grid[None, :]) ** 2
    interacting = ordered_sector_hamiltonian(
        one_body, 2, pair_potential=potential
    )
    noninteracting = ordered_sector_hamiltonian(one_body, 2)
    diagonal_shift = torch.diagonal(interacting - noninteracting)
    assert torch.all(diagonal_shift >= 0)
    assert torch.count_nonzero((interacting - noninteracting) - torch.diag(diagonal_shift)) == 0


def test_ordered_soft_coulomb_equals_dense_exterior_truth() -> None:
    grid, one_body = finite_difference_harmonic_hamiltonian(7, 0.8)
    potential = 1 / torch.sqrt((grid[:, None] - grid[None, :]) ** 2 + 1)
    identity = torch.eye(7, dtype=one_body.dtype)
    dense_two_body = torch.einsum(
        "pq,pr,qs->pqrs", potential, identity, identity
    )
    ordered = ordered_sector_hamiltonian(
        one_body, 3, pair_potential=potential
    )
    exterior = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, 3, dense_two_body
    )
    torch.testing.assert_close(ordered, exterior, atol=2e-13, rtol=2e-13)
