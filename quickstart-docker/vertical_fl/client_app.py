import torch

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

from torch import Tensor
from torch.optim import SGD
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from vertical_fl.task import ClientModel, load_data

from vertical_fl.utils import set_seed, GLOBAL_SEED

# Directory for saving/loading client checkpoints
CHECKPOINT_DIR = Path(__file__).parent.parent / "model" / "clients"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class FlowerClient(NumPyClient):
    """
    Vertical-FL client computing local embeddings and applying gradients.
    """
    def __init__(
        self,
        v_split_id: int,
        data: Tensor,
        lr: float,
    ) -> None:
        self.v_split_id = v_split_id
        self.data = data
        set_seed(GLOBAL_SEED + v_split_id)  # Ensure reproducibility
        self.model = ClientModel(input_size=self.data.shape[1])
        # Load checkpoint if exists
        ckpt = CHECKPOINT_DIR / f"client_{self.v_split_id}.pth"
        if ckpt.exists():
            state = torch.load(ckpt)
            self.model.load_state_dict(state)
        self.optimizer = SGD(self.model.parameters(), lr=lr)

    def get_parameters(
        self,
        config: Dict[str, Any],
    ) -> List[Any]:
        # Not used in VFL, but required by interface
        return []

    def fit(
        self,
        parameters: List[Any],
        config: Dict[str, Any],
    ) -> Tuple[List[Any], int, Dict[str, float]]:
        """
        Forward pass to compute embeddings and save client model.

        Returns:
            - List of embeddings for server aggregation
            - Number of examples used
            - Metrics dict (empty)
        """

        rnd = int(config.get("round", 0))
        set_seed(GLOBAL_SEED + rnd + self.v_split_id)  # Ensure reproducibility
        torch.use_deterministic_algorithms(True, warn_only=True)
        # print(f"\n\n!!!!!   Client {self.v_split_id} on round {rnd}, setting seed to {GLOBAL_SEED + rnd + self.v_split_id}  !!!!!\n\n")


        embedding = self.model(self.data)

        # Save updated client model
        ckpt = CHECKPOINT_DIR / f"client_{self.v_split_id}.pth"
        torch.save(self.model.state_dict(), ckpt)
        return [embedding.detach().numpy()], len(self.data), {"v_id": self.v_split_id}

    def evaluate(
        self,
        parameters: List[Any],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Apply server-sent gradient to local model.

        Returns dummy loss (0.0) and number of examples.
        """

        rnd = int(config.get("round", 0))
        set_seed(GLOBAL_SEED + rnd)  # Ensure reproducibility
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.model.zero_grad()
        embedding = self.model(self.data)
        grad = torch.from_numpy(parameters[self.v_split_id])
        embedding.backward(grad)
        self.optimizer.step()
        return 0.0, len(self.data), {}


def client_fn(context: Context) -> NumPyClient:
    """
    Build FlowerClient from runtime context.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Load and preprocess data
    df_partition, v_split_id = load_data(partition_id, num_partitions)

    set_seed(GLOBAL_SEED + v_split_id)  # Ensure reproducibility

    scaled = StandardScaler().fit_transform(df_partition)
    data = torch.tensor(scaled).float()
    lr = float(context.run_config.get("learning-rate", 0.01))
    return FlowerClient(v_split_id, data, lr).to_client()

# Launch the Flower client
app = ClientApp(client_fn=client_fn)
