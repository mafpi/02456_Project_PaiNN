import torch

class AtomNeighbours:
    """
    A class to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance
    in each graph of a batch. (One atom is not neighbour of itself), (Neighbors are only in the same molelcule)
    """
    
    def __init__(self, cutoff:float):
        """
        Initializes the AtomNeighbours class with a specified distance cutoff.

        Parameters:
        - cutoff (float): The maximum distance within which two atoms are considered neighbors.
        """
        self.cutoff = cutoff

    def neigbourhood_matrix(self, atom_positions:torch.Tensor, graph_indexes:torch.Tensor) -> torch.Tensor: 
        """
        Calculates the neighborhood adjacency matrix for a batch of atomic graphs.

        Parameters:
        - batch: A batch of atomic graphs, containing:
            - batch.pos (torch.Tensor): Tensor of atomic positions with shape (num_atoms, 3).
            - batch.batch (torch.Tensor): Tensor indicating which graph each atom belongs to, 
                                           with shape (num_atoms,).

        Returns:
        - torch.Tensor: A matrix where each row represents a pair of neighboring atoms
                        within the cutoff distance. Each row contains:
                        [atom_i_index, atom_j_index, x_diff, y_diff, z_diff, distance].

        Example output:
        If the neighborhood matrix contains two pairs of neighbors within the cutoff distance, 
        the return value might look like this:
            tensor([[0, 3, -0.5, -0.5, -0.5, 0.8660],
                    [3, 0,  0.5,  0.5,  0.5, 0.8660]])
        """

        # Step 0: Check that batch.pos and batch.batch are torch.Tensors #TODO i think it is way more elegant that the arguments of neigbourhood_matrix are batch.pos:torch.Tensor and batch.batch:torch.Tensor 
        if not isinstance(atom_positions, torch.Tensor):
            raise TypeError("batch.pos must be a torch.Tensor")
        if not isinstance(graph_indexes, torch.Tensor):
            raise TypeError("batch.batch must be a torch.Tensor")
        
        # Step 1: Create an adjacency matrix indicating if two atoms belong to the same graph.
        # adj_matrix[i, j] = 1 if atoms i and j are in the same graph, else 0.
        adj_matrix = (graph_indexes.unsqueeze(1) == graph_indexes.unsqueeze(0)).int()
        
        # Step 2: Set the diagonal to 0 because an atom cannot be its own neighbor.
        adj_matrix.fill_diagonal_(0)

        # Step 3: Find indices (i, j) where atoms are in the same graph (adj_matrix[i, j] == 1).
        idx_i, idx_j = torch.nonzero(adj_matrix, as_tuple=True)

        # Step 4: Calculate the pairwise distances between atoms i and j in the same graph.
        x_ij = atom_positions[idx_i, 0] - atom_positions[idx_j, 0]  # x-coordinate difference (x_i - x_j)
        y_ij = atom_positions[idx_i, 1] - atom_positions[idx_j, 1]  # y-coordinate difference (y_i - y_j)
        z_ij = atom_positions[idx_i, 2] - atom_positions[idx_j, 2]  # z-coordinate difference (z_i - z_j)
        d_ij = torch.sqrt(x_ij**2 + y_ij**2 + z_ij**2)    # Euclidean distance between atoms i and j

        # Step 5: Stack the results into a matrix where each row corresponds to:
        # [atom_i_index, atom_j_index, x_ij, y_ij, z_ij, d_ij]
        neig_matrix = torch.stack([idx_i.to(torch.int), idx_j.to(torch.int), x_ij, y_ij, z_ij, d_ij], dim=1)

        # Step 6: Filter out atom pairs where the distance is greater than the cutoff.
        neig_matrix = neig_matrix[d_ij <= self.cutoff]

        return neig_matrix

