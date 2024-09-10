from futils import snapshot
from bimuma.summary import (
    BinaryMutationMatrix,
    Correction,
    Sfs,
    compute_burden,
    compute_sfs,
    compute_variants,
)


class Donor:
    """A donor stores the site frequency spectrum (SFS) and the mutational
    burden computed from the binary matrix.
    """

    def __init__(self, binary_mut: BinaryMutationMatrix):
        self.cells = binary_mut.cells
        self.polytomies = binary_mut.polytomies
        self.burden = compute_burden(binary_mut)
        self.sfs = compute_sfs(binary_mut)
        self.corrected_variants_one_over_1_squared = None

    def _sampled_sfs(self, pop_size: int, sample_size: int):
        self.corrected_variants_one_over_1_squared = compute_variants(
            Correction.ONE_OVER_F_SQUARED,
            pop_size=pop_size,
            sample_size=sample_size,
        )

    def sfs_1_over_f_squared_corrected(self, pop_size: int, sample_size: int) -> Sfs:
        if self.corrected_variants_one_over_1_squared is None:
            self.corrected_variants_one_over_1_squared = self._sampled_sfs(
                pop_size, sample_size
            )
        cells = self.corrected_variants_one_over_1_squared.corrected_variants.shape[0]
        x = self.corrected_variants_one_over_1_squared.frequencies[:cells]
        f_sampled = self.corrected_variants_one_over_1_squared.corrected_variants
        return Sfs(snapshot.Histogram({xx: f for xx, f in zip(x, f_sampled)}))
