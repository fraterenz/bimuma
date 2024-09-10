"""The site frequency spectrum (SFS) and the mutational burden computed from
the binary mutation matrix.

TODO: explain SFS and burden
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from futils import snapshot
from typing import List, NewType, Set, Tuple
from scipy import stats
import numpy as np
import pandas as pd

Sfs = NewType("Sfs", snapshot.Histogram)
Burden = NewType("Burden", snapshot.Histogram)


class NonUniqueMutations(ValueError):
    """Non unique mutations found in the binary matrix"""


class NonUniqueCells(ValueError):
    """Non unique cells found in the binary matrix"""


class BinaryMutationMatrix:
    def __init__(self, matrix: pd.DataFrame) -> None:
        """A binary mutation matrix where rows (index) are unique mutations and
        columns are cells."""

        def get_unique_quantities(quantities: List[str]) -> Set[str]:
            unique_q = set(quantities)
            assert len(unique_q) == len(quantities)
            return unique_q

        assert matrix.shape[0], "empty matrix"
        self.matrix = matrix
        try:
            self.cells = get_unique_quantities(matrix.columns.to_list())
        except AssertionError:
            raise NonUniqueCells
        try:
            self.mutations = get_unique_quantities(matrix.index.to_list())
        except AssertionError:
            raise NonUniqueMutations
        # polytomy or artefact: different cells with the same genotype
        self.polytomies = matrix.T.duplicated(keep=False).sum()

    def get_mutations(self) -> Set[str]:
        return self.mutations

    def get_cells(self) -> Set[str]:
        return self.cells


def binary_mut_mat_from_excel(
    path2excel: Path, sheet_name: int = 0, index_col: int = 0
) -> BinaryMutationMatrix:
    data = pd.read_excel(
        path2excel,
        sheet_name=sheet_name,
        index_col=index_col,
    )
    return BinaryMutationMatrix(data)


def filter_cells_from_matrix(binary_mut: BinaryMutationMatrix, cells2keep: Set[str]) -> BinaryMutationMatrix:
    data = binary_mut.matrix
    data = data.drop(columns=data.loc[:, ~data.columns.isin(cells2keep)].columns)
    assert data.shape[1], "Dropped all cells"
    return BinaryMutationMatrix(data)


def compute_sfs(binary_mut: BinaryMutationMatrix) -> Sfs:
    # the frequency of mutations in the matrix (sum on axis=1) grouped into
    # occurences by value_counts
    sfs_donor = binary_mut.matrix.sum(axis=1).value_counts()
    # drop mutations non occurring in any cell
    sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)
    x_sfs = sfs_donor.index.to_numpy(dtype=int)
    return Sfs(
        snapshot.histogram_from_dict(
            {x: y for x, y in zip(x_sfs, sfs_donor.to_numpy())}
        )
    )


def compute_burden(binary_mut: BinaryMutationMatrix) -> Burden:
    burden = binary_mut.matrix.sum(axis=0).value_counts()
    x_burden = burden.index.to_numpy(dtype=int)
    return Burden(
        snapshot.histogram_from_dict(
            {x: y for x, y in zip(x_burden, burden.to_numpy())}
        )
    )


def average_burden(burdens: List[snapshot.Histogram]):
    burden_uniformised = snapshot.Uniformise.uniformise_histograms(
        [snapshot.Histogram(burden) for burden in burdens]
    )
    jcells = burden_uniformised.create_x_array()
    avg_burden = burden_uniformised.y
    # compute the average, not pooling
    return jcells, np.mean(avg_burden, axis=0)


def pooled_burden(burdens: List[Burden]) -> snapshot.Distribution:
    return snapshot.Uniformise.pooled_distribution(
        [snapshot.Histogram(b) for b in burdens]
    )


def compute_burden_mean_variance(burden: Burden) -> Tuple[float, float]:
    cells = sum(burden.values())
    mean = sum(map(lambda entry: entry[0] * entry[1] / cells, burden.items()))
    variance = sum(
        map(
            lambda entry: (entry[0] - mean) ** 2 * entry[1] / cells,
            burden.items(),
        )
    )
    return mean, variance


class Correction(Enum):
    ONE_OVER_F = auto()
    ONE_OVER_F_SQUARED = auto()


def compute_frequencies(pop_size: int) -> np.ndarray:
    return np.arange(1, pop_size + 1, step=1, dtype=int)


def compute_sampling_correction(n: int, s: int) -> np.ndarray:
    return np.array(
        [
            stats.binom(s, (v + 1) / n).pmf(np.arange(1, s + 1, step=1))
            for v in range(0, n)
        ],
        dtype=float,
    ).T


@dataclass
class CorrectedVariants:
    correction: Correction
    corrected_variants: np.ndarray
    variant2correct: np.ndarray
    frequencies: np.ndarray


def compute_variants(
    correction: Correction, pop_size: int, sample_size: int
) -> CorrectedVariants:
    """
    _f = range(0,1,length=N+1)
    nThGr_f = (1 ./ _f).^2
    nSample_f = Vector{Float64}(undef, S+1)
    for u in 1:S
        nSample_f[1+u] =
            sum(
                [ nTrue_f[1+v] * pdf(Binomial(S, v/N), u) for v=1:N ]
            )
    end
    """
    frequencies = compute_frequencies(pop_size)
    if correction == Correction.ONE_OVER_F:
        variants2correct = 1 / frequencies
    elif correction == Correction.ONE_OVER_F_SQUARED:
        variants2correct = 1 / frequencies**2
    else:
        raise ValueError
    assert variants2correct.shape[0] == pop_size, f"{variants2correct.shape[0]}"
    corrected = (
        compute_sampling_correction(pop_size, sample_size)[:sample_size, :]
        @ variants2correct
    )
    return CorrectedVariants(correction, corrected, variants2correct, frequencies)
