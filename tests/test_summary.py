from io import StringIO
import pytest
import numpy as np
import pandas as pd
from src.bimuma.summary import (
    BinaryMutationMatrix,
    NonUniqueCells,
    NonUniqueMutations,
    filter_cells_from_matrix,
)


def create_binary_mut() -> BinaryMutationMatrix:
    data = """PD43947h_lo0002_hum\tPD43947h_lo0003_hum\tPD43947h_lo0004_hum\tPD43947h_lo0005_hum
10-110257574-G-A\t0.5\t1\t0\t0
10-114594383-C-T\t0\t1\t0\t0
10-124679941-T-C\t0\t0\t0\t0
10-12684595-G-C\t0\t0\t1\t1
10-132754174-G-A\t0\t0\t0\t0
10-133246550-G-A\t0\t1\t0\t0
10-133335354-C-T\t0\t0\t0\t0
10-134888630-G-A\t0\t1\t1\t1
10-15817209-T-C\t0\t0\t1\t1
10-1937569-G-C\tNaN\t0\t0\t0
 """
    return BinaryMutationMatrix(pd.read_csv(StringIO(data), sep="\t"))


def test_empty_matrix():
    with pytest.raises(AssertionError):
        BinaryMutationMatrix(pd.DataFrame())


def test_nan_0_dot_5_values():
    binary_mut = create_binary_mut()
    assert binary_mut.matrix.select_dtypes("int").shape == binary_mut.matrix.shape
    assert not binary_mut.matrix.isna().any().any()
    assert np.all(np.ones(shape=binary_mut.matrix.shape) - binary_mut.matrix.to_numpy() <= 1)
    assert binary_mut.matrix.iloc[0, 0] == 0

def test_duplicated_cells():
    binary_mut = create_binary_mut()
    binary_mut = pd.concat([binary_mut.matrix, binary_mut.matrix.iloc[:2]], axis=1)
    with pytest.raises(NonUniqueCells):
        BinaryMutationMatrix(binary_mut)


def test_duplicated_mutations():
    binary_mut = create_binary_mut()
    binary_mut = pd.concat([binary_mut.matrix, binary_mut.matrix.iloc[:2]], axis=0)
    with pytest.raises(NonUniqueMutations):
        BinaryMutationMatrix(binary_mut)


def test_polytomies():
    assert create_binary_mut().polytomies == 2


def test_filter_cells():
    binary_mut = create_binary_mut()
    cells2keep = set(binary_mut.matrix.columns[:2].to_list())
    filtered = filter_cells_from_matrix(binary_mut, cells2keep)
    assert len(set(filtered.matrix.columns.to_list()) - cells2keep) == 0
