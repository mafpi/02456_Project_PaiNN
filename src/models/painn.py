import torch
import torch.nn as nn
import data.AtomNeighbours as AN

class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()

        self.scalar_embedding = nn.Embedding(num_unique_atoms, num_features)

        self.message_layer = nn.ModuleList(
            [MessagePaiNN(num_features, num_rbf_features, cutoff_dist) for _ in range(num_message_passing_layers)]
        )

        self.update_layer = nn.ModuleList(
            [UpdatePaiNN(num_features) for _ in range(num_message_passing_layers)]
        )

        self.last_layer = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features),
        )
        
        self.AN = AN.AtomNeighbours(cutoff_dist)


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        # Neighbourhood matrix
        self.adj_matrix = self.AN.neigbourhood_matrix(atom_positions, graph_indexes)

        node_scalar = self.scalar_embedding(atoms)
        node_vector = torch.zeros(atoms.size(0), self.num_features, 3)

        for message_layer, update_layer in zip(self.message_layer, self.update_layer):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, self.adj_matrix)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)

        node_scalar = self.last_layer(node_scalar)

        output = torch.zeros(atoms.size(0), self.num_outputs)
        output.index_add_(0, self.adj_matrix[:, 0], node_scalar)

        return output


def sinc_expansion(r_ij: torch.Tensor, n: int, cutoff: float):

    n_vals = torch.arange(n) + 1

    return torch.sin(r_ij.unsqueeze(-1) * n_vals * torch.pi / cutoff) / r_ij.unsqueeze(-1)


def cosine_cutoff(r_ij: torch.Tensor, cutoff: float):
    return torch.where(
        r_ij < cutoff,
        0.5 * (torch.cos(torch.pi * r_ij / cutoff) + 1),
        torch.tensor(0.0),
    )

class MessagePaiNN(nn.Module):
    """
    Message passing.
    """
    def __init__(
        self,
        # num_message_passing_layers: int = 3,
        num_features: int = 128,
        # num_outputs: int = 1,
        num_rbf_features: int = 20,
        # num_unique_atoms: int = 100,
        # cutoff_dist: float = 5.0,
    ) -> None:
        # """
        # Args:
        #     num_message_passing_layers: Number of message passing layers in
        #         the PaiNN model.
        #     num_features: Size of the node embeddings (scalar features) and
        #         vector features.
        #     num_outputs: Number of model outputs. In most cases 1.
        #     num_rbf_features: Number of radial basis functions to represent
        #         distances.
        #     num_unique_atoms: Number of unique atoms in the data that we want
        #         to learn embeddings for.
        #     cutoff_dist: Euclidean distance threshold for determining whether 
        #         two nodes (atoms) are neighbours.
        # """
        super().__init__()

        self.scalar_message = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, 3 * self.num_features),
        )

        self.layer_rbf = nn.Linear(self.num_rbf_features, 3* self.num_features)


    def forward(
        self,
        node_scalar,
        node_vector,
        adj_matrix
    ) -> torch.FloatTensor:
        """
        xxxx

        Args:
            djfvkd:ajcnac

        Returns:
            XXXXX
        """
        
        atom_scalar = self.scalar_message(node_scalar[adj_matrix[:, 1]])

        # RBF

        r_ij = adj_matrix[:, 2:5] 
        rbf = self.layer_rbf(sinc_expansion(r_ij, self.num_rbf_features, self.cutoff_dist)) 
        rbf_cos_cutoff = rbf * cosine_cutoff(r_ij, self.cutoff_dist).unsqueeze(-1)

        pre_split = atom_scalar * rbf_cos_cutoff

        # Split
        split1, split2, split3 = torch.split(pre_split, self.num_features, dim = 1)

        r_ij_standardized = r_ij / adj_matrix[:, 5] 
        message_edge = split3.unsqueeze(-1) * r_ij_standardized.unsqueeze(-1)
        message_vector = node_vector * split1.unsqueeze(-1) + message_edge

        delta_v = torch.zeros_like(node_vector)
        delta_s = torch.zeros_like(node_scalar)

        # list_neighbours: index of the neighbours of atom i
        delta_s.index_add_(0, adj_matrix[:, 0], split2)
        delta_v.index_add_(0, adj_matrix[:, 0], message_vector)        

        return node_scalar + delta_s, node_vector + delta_v
    



class UpdatePaiNN(nn.Module):
    """
    Update passing.
    """
    def __init__(
        self,
        # num_message_passing_layers: int = 3,
        num_features: int = 128,
        # num_outputs: int = 1,
        # num_rbf_features: int = 20,
        # num_unique_atoms: int = 100,
        # cutoff_dist: float = 5.0,
    ) -> None:
        super().__init__()

        self.update_U = nn.Linear(self.num_features, self.num_features, bias=False)
        self.update_V = nn.Linear(self.num_features, self.num_features, bias=False)

        self.scalar_update = nn.Sequential(
            nn.Linear(self.num_features * 2, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, 3 * self.num_features),
        )


    def forward(
        self,
        node_scalar,
        node_vector
    ) -> torch.FloatTensor:
        """
        xxxx

        Args:
            djfvkd:ajcnac

        Returns:
            XXXXX
        """

        U = self.update_U(node_vector)
        V = self.update_V(node_vector)

        V_norm = torch.norm(V, dim = -1)
        print(V_norm.shape)

        pre_split_s = self.scalar_update(torch.cat((V_norm, node_scalar), dim = 1))

        a_vv, a_sv, a_ss = torch.split(pre_split_s, self.num_features, dim = 1)

        delta_v = a_vv * U

        inner_prod = torch.sum(U * V, dim=1)
        delta_s = inner_prod * a_sv + a_ss  

        return node_scalar + delta_s, node_vector + delta_v