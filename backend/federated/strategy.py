"""
Flower-compatible federated strategy for AFSPL.
Extends FedAvg with sparse prompt reconstruction.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Scalar, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger


class AFSPLFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower strategy extending FedAvg with:
    - Sparse top-K prompt reconstruction
    - Round-wise metric logging
    - Early stopping support
    """

    def __init__(
        self,
        prompt_length: int = 16,
        hidden_dim: int = 512,
        k_ratio: float = 0.3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.k_ratio = k_ratio
        self.round_metrics: list[dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates using sparse FedAvg.
        Each client sends: (indices, values) packed into numpy arrays.
        """
        if not results:
            return None, {}

        # Collect weights and sparse updates
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        # Initialize aggregation buffer
        token_sum = np.zeros((self.prompt_length, self.hidden_dim), dtype=np.float32)
        token_weight = np.zeros(self.prompt_length, dtype=np.float32)

        for _, fit_res in results:
            w = fit_res.num_examples / total_examples
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) >= 2:
                indices = arrays[0].astype(int)    # (K,)
                values = arrays[1]                  # (K, H)
                for i, idx in enumerate(indices):
                    if 0 <= idx < self.prompt_length:
                        token_sum[idx] += w * values[i]
                        token_weight[idx] += w

        # Reconstruct: only update received tokens
        # For unreceived tokens we return zeros and handle on client side
        aggregated = np.zeros_like(token_sum)
        mask = token_weight > 0
        aggregated[mask] = token_sum[mask] / token_weight[mask, np.newaxis]

        metrics_agg = {
            "server_round": float(server_round),
            "num_clients": float(len(results)),
            "token_coverage": float(mask.sum()),
        }
        self.round_metrics.append({"round": server_round, **metrics_agg})
        logger.info(f"Round {server_round}: aggregated {len(results)} clients, coverage={mask.sum()}/{self.prompt_length}")

        return ndarrays_to_parameters([aggregated, mask.astype(np.float32)]), metrics_agg

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {"server_round": server_round, "k_ratio": self.k_ratio}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=max(1, int(client_manager.num_available() * self.fraction_fit)),
            min_num_clients=self.min_fit_clients,
        )
        return [(c, fit_ins) for c in clients]
