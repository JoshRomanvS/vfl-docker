import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from vertical_fl.utils import set_seed, GLOBAL_SEED

# Directory for saving/loading server checkpoints
CHECKPOINT_DIR = Path(__file__).parent.parent / "model" / "central"
class ServerModel(nn.Module):
    """
    A simple server-side model that aggregates client embeddings.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(x))

class Strategy(fl.server.strategy.FedAvg):
    """
    Custom Flower strategy for vertical federated learning with 3 clients.
    - Loads existing server checkpoint if available.
    - Aggregates embeddings via a simple linear model + BCE loss.
    - Saves new checkpoint after aggregation.
    """
    def __init__(
        self,
        labels: List[float],
        lr: float = 0.01,
        **kwargs,
    ) -> None:
        # Ensure checkpoint directory exists
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Set global seed for reproducibility
        set_seed(GLOBAL_SEED)

        # Initialize server model
        self.model = ServerModel(12)
        # Attempt to load latest checkpoint
        latest_ckpt = self._get_latest_checkpoint()
        if latest_ckpt:
            state = torch.load(latest_ckpt)
            self.model.load_state_dict(state)
            print(f"Loaded server checkpoint: {latest_ckpt}")

        # Prepare initial parameters for Flower
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(params)

        # Initialize base FedAvg with strict client requirements
        super().__init__(
            initial_parameters=initial_parameters,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_available_clients=3,
            min_evaluate_clients=3,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            **kwargs,
        )

        # Optimizer and loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        # Prepare server labels tensor
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[str, fl.server.client_proxy.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, float]]:
        # Skip aggregation on failures if not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Ensure reproducability
        set_seed(GLOBAL_SEED + rnd)
        print(f"\n\n!!!!!   In aggregate_fit: Round {rnd}, used seed: {GLOBAL_SEED + rnd}  !!!!!\n\n")

        client_embeddings = [parameters_to_ndarrays(res.parameters)[0] for _, res in results]
        embedding_shapes = [emb.shape for emb in client_embeddings]
        embedding_sums = [emb.sum(axis=0) for emb in client_embeddings]

        # print(f"Aggregating {len(client_embeddings)} client embeddings with shapes: {embedding_shapes} and sum: {embedding_sums}")

        # Collect and concatenate client embeddings
        embedding_list = [
            torch.from_numpy(emb) for emb in client_embeddings
        ]
        embeddings = torch.cat(embedding_list, dim=1)
        embeddings = embeddings.detach().requires_grad_()

        # Forward + backward on server model
        outputs = self.model(embeddings)
        loss = self.criterion(outputs, self.labels)
        loss.backward()

        # Server update
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Split gradients back to clients
        grads = embeddings.grad.split([4, 4, 4], dim=1)
        np_grads = [g.numpy() for g in grads]
        aggregated_parameters = ndarrays_to_parameters(np_grads)

        gradient_shapes = [g.shape for g in np_grads]
        gradient_sums = [g.sum(axis=0) for g in np_grads]

        # Compute accuracy metric
        with torch.no_grad():
            preds = (self.model(embeddings) > 0.5).float()
            correct = (preds == self.labels).sum().item()
            accuracy = correct / len(self.labels) * 100

        # Save new checkpoint
        self._save_checkpoint()

        return aggregated_parameters, {"accuracy": accuracy, "embedding_shapes": embedding_shapes, "embedding_sums": embedding_sums, "gradient_shapes": gradient_shapes, "gradient_sums": gradient_sums}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[str, fl.server.client_proxy.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        # No server-side evaluation
        return None, {}

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """
        Return the most recent checkpoint file, if any.
        """
        files = list(CHECKPOINT_DIR.glob('checkpoint_*.pth'))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def _save_checkpoint(self) -> None:
        """
        Save the server model state to a new checkpoint file.
        """
        timestamp = int(time.time())
        ckpt_path = CHECKPOINT_DIR / f"checkpoint_{timestamp}.pth"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Saved server checkpoint: {ckpt_path}")
