import torch
import unittest
from AtomNeighbours import AtomNeighbours
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)


class AtomNeighboursTests(unittest.TestCase):
    """
    Unit tests for the AtomNeighbours class.

    The following tests check the behavior of AtomNeighbours.neighbourhood_matrix on:
    - A saved batch (loaded from a file) to ensure that the method correctly identifies neighboring atoms
      in a real-world scenario with predefined data.
    - A manually created batch with known values, which allows for precise validation. In this test, we 
      manually calculate the expected result, allowing us to verify the method's correctness by comparing 
      its output against our manually derived expected results.

    For each test, several checks are performed:
    - Ensuring that only atoms within the same molecule are identified as neighbors.
    - Ensuring that only atom pairs within the specified cutoff distance are considered as neighbors.
    - Verifying that the indices of the neighbors are in ascending order.
    - Confirming that each neighbor pair (idx_i, idx_j) has a corresponding symmetric pair (idx_j, idx_i).
    """

    def setUp(self):
        """
        Sets up test data for use in multiple tests.

        - Defines a small batch with atomic positions and graph identifiers for a manually created batch.
        - This batch is used to verify if AtomNeighbours correctly identifies neighboring atoms with a given cutoff.
        - The expected adjacency data is manually calculated for a cutoff distance of 1.5 to verify the method's accuracy.
        """
        # Define a small test batch object with required attributes for general testing
        class TestBatch:
            def __init__(self, pos, batch):
                self.pos = pos
                self.batch = batch

        # Example atomic positions and graph identifiers
        self.atomic_positions = torch.tensor([
            [0.0, 0.0, 0.0],  # Atom 0, Molecule 1
            [1.0, 1.0, 1.0],  # Atom 1, Molecule 1
            [2.0, 2.0, 2.0],  # Atom 2, Molecule 2 (different molecule)
            [0.5, 0.5, 0.5]   # Atom 3, Molecule 1
        ], dtype=torch.float)

        self.graph_indexes = torch.tensor([0, 0, 1, 0])  # Molecule identifiers

        # Instantiate the test batch for the small test case
        self.batch = TestBatch(pos=self.atomic_positions, batch=self.graph_indexes)

        # Manually calculated expected adjacency data for validation with a cutoff of 1.5
        # Only pairs within the same molecule and within the cutoff distance are included
        self.expected_data_rows = [
            [0, 3, -0.5, -0.5, -0.5, 0.8660254037844386],
            [1, 3,  0.5,  0.5,  0.5, 0.8660254037844386],
            [3, 0,  0.5,  0.5,  0.5, 0.8660254037844386],
            [3, 1, -0.5, -0.5, -0.5, 0.8660254037844386]
        ]

    def _run_general_checks(self, adjacency_data_torch, batch, cutoff):
        """
        Helper function to perform common validation checks on adjacency data.

        Parameters:
        - adjacency_data_torch (torch.Tensor): The adjacency data produced by AtomNeighbours.
        - batch (TestBatch): The batch data containing atom positions and graph identifiers.
        - cutoff (float): The cutoff distance used for finding neighbors.

        Checks performed:
        - Ensures that neighboring atom pairs belong to the same molecule.
        - Verifies that all neighbor distances are within the specified cutoff.
        - Confirms that indices are in ascending order.
        - Ensures each neighbor pair has a symmetric counterpart.
        """
        adjacency_data_list = adjacency_data_torch.tolist()

        # Check that pairs of atoms from different molecules are not included
        for row in adjacency_data_list:
            atom_i, atom_j = int(row[0]), int(row[1])
            self.assertEqual(batch.batch[atom_i], batch.batch[atom_j],
                             f"Different molecule check failed for pair: {atom_i}-{atom_j}")

        # Check that pairs of atoms with distance > cutoff are not included
        for row in adjacency_data_list:
            distance_i_j = row[-1]  # Distance is now the last column
            self.assertLessEqual(distance_i_j, cutoff,
                                 f"Cutoff distance check failed for distance: {distance_i_j}")

        # Check that idx_i values are in ascending order
        idx_i_values = [row[0] for row in adjacency_data_list]
        self.assertEqual(idx_i_values, sorted(idx_i_values),
                         f"Index order check failed! Expected ascending order for idx_i values: {idx_i_values}")

        # Check that each (idx_i, idx_j) pair has a corresponding (idx_j, idx_i) pair
        adjacency_pairs = {(int(row[0]), int(row[1])) for row in adjacency_data_list}
        for (idx_i, idx_j) in adjacency_pairs:
            self.assertIn((idx_j, idx_i), adjacency_pairs,
                          f"Symmetry check failed! Pair ({idx_i}, {idx_j}) does not have a matching ({idx_j}, {idx_i})")

    def test_neighbourhood_matrix_with_predefined_data(self):
        """
        Test AtomNeighbours.neighbourhood_matrix with a manually defined batch and expected data.
        
        - Creates an instance of AtomNeighbours with a cutoff of 1.5.
        - Validates the method's output against manually calculated expected data.
        - Runs additional general checks (same molecule, cutoff distance, ascending order, symmetry) to verify correctness.
        """
        cutoff = 1.5
        atom_neighbours = AtomNeighbours(cutoff)
        adjacency_data_torch = atom_neighbours.neigbourhood_matrix(self.batch)

        # Convert expected data to tensor and compare
        expected_adjacency_data_torch = torch.tensor(self.expected_data_rows, dtype=torch.float)
        self.assertTrue(torch.allclose(adjacency_data_torch, expected_adjacency_data_torch, atol=1e-13),
                        f"Expected:\n{expected_adjacency_data_torch}\nGot:\n{adjacency_data_torch}")

        # Run the general checks
        self._run_general_checks(adjacency_data_torch, self.batch, cutoff)

    def test_neighbourhood_matrix_with_saved_batch(self):
        """
        Test AtomNeighbours.neighbourhood_matrix with a saved batch loaded from file.

        - Loads a previously saved batch from the "./Tests/example_batch.pth" file.
        - Creates an instance of AtomNeighbours with a cutoff of 1.5.
        - Runs general checks (same molecule, cutoff distance, ascending order, symmetry) to ensure the method 
          behaves correctly with real-world data.
        """
        # Load a saved batch for testing if needed
        saved_batch_path = "./Test_helper/example_batch.pth"
        batch = torch.load(saved_batch_path)

        # Create an instance of AtomNeighbours with a cutoff of 1.5
        cutoff = 1.5
        atom_neighbours = AtomNeighbours(cutoff)
        adjacency_data_torch = atom_neighbours.neigbourhood_matrix(batch)

        # Run the general checks
        self._run_general_checks(adjacency_data_torch, batch, cutoff)


if __name__ == '__main__':
    unittest.main()
