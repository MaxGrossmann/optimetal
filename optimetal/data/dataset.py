from typing import Any
from torch import Tensor
from torch_geometric.data import Data

class ThreebodyData(Data):
    """
    Custom torch_geometric 'Data' object to ensure correct batching behavior for edge (twobody) indices and 
    angle (threebody) indices in my dataset, see https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html.
    """
    
    def __inc__(
        self, 
        key: str, 
        value: Tensor,
        *args: Any, 
        **kwargs: Any,
    ) -> int:
        if key == "edge_index":
            return self.num_nodes
        if key == "threebody_index":
            return self.num_twobody
        return super().__inc__(key, value, *args, **kwargs)

