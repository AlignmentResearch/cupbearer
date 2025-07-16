import logging
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from nflows import distributions, flows, transforms
from torch import nn

from cupbearer import utils
from cupbearer.detectors.feature_model.feature_model_detector import FeatureModel, FeatureModelDetector

logger = logging.getLogger(__name__)


class NormalizingFlow(nn.Module):
    """Normalizing Flow wrapper for anomaly detection.

    This class wraps nflows library to provide a simple interface for
    anomaly detection using normalizing flows.
    """

    def __init__(
        self,
        input_dim: int,
        flow_type: str = "maf",
        n_layers: int = 5,
        hidden_dim: int = 512,
        n_blocks: int = 2,
        dtype: torch.dtype = torch.float32,
        use_batch_norm: bool = True,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.flow_type = flow_type
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dtype = dtype

        # Create the flow based on the specified type
        self.flow = self._create_flow(
            flow_type=flow_type,
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            use_batch_norm=use_batch_norm,
            dropout_probability=dropout_probability,
            dtype=dtype,
        )
        self.flow = self.flow.to(self.dtype)

    def _create_flow(
        self,
        flow_type: str,
        input_dim: int,
        n_layers: int,
        hidden_dim: int,
        n_blocks: int,
        use_batch_norm: bool,
        dropout_probability: float,
        dtype: torch.dtype,
    ) -> flows.Flow:
        """Create a normalizing flow based on the specified type."""

        # Base distribution (standard normal)
        base_dist = distributions.StandardNormal(shape=[input_dim])

        # Create transform layers
        transform_layers = []

        if flow_type.lower() == "maf":
            # Masked Autoregressive Flow
            for _ in range(n_layers):
                transform_layers.append(
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=input_dim,
                        hidden_features=hidden_dim,
                        num_blocks=n_blocks,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=nn.functional.relu,
                        dropout_probability=dropout_probability,
                        use_batch_norm=use_batch_norm,
                    )
                )
                # Add random permutation between layers
                transform_layers.append(transforms.RandomPermutation(features=input_dim))

        elif flow_type.lower() == "realnvp":
            # Real NVP
            def create_transform_net(in_features, out_features):
                """Create a transform network that handles context properly."""
                net = nn.Sequential(
                    nn.Linear(in_features, hidden_dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_features, dtype=dtype),
                )

                # Wrap the network to handle context parameter
                class ContextIgnoringNetwork(nn.Module):
                    def __init__(self, network):
                        super().__init__()
                        self.network = network

                    def forward(self, x, context=None):
                        return self.network(x)

                return ContextIgnoringNetwork(net)

            for i in range(n_layers):
                # Alternating mask pattern
                mask = torch.arange(input_dim) % 2 == (i % 2)
                transform_layers.append(
                    transforms.AffineCouplingTransform(
                        mask=mask,
                        transform_net_create_fn=create_transform_net,
                    )
                )
                # Add random permutation between layers
                if i < n_layers - 1:
                    transform_layers.append(transforms.RandomPermutation(features=input_dim))

        elif flow_type.lower() == "coupling":
            # Coupling layers (similar to RealNVP but with different coupling functions)
            def create_coupling_transform_net(in_features, out_features):
                """Create a transform network that handles context properly."""
                net = nn.Sequential(
                    nn.Linear(in_features, hidden_dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_features, dtype=dtype),
                )

                # Wrap the network to handle context parameter
                class ContextIgnoringNetwork(nn.Module):
                    def __init__(self, network):
                        super().__init__()
                        self.network = network

                    def forward(self, x, context=None):
                        return self.network(x)

                return ContextIgnoringNetwork(net)

            for i in range(n_layers):
                mask = torch.arange(input_dim) % 2 == (i % 2)
                transform_layers.append(
                    transforms.PiecewiseRationalQuadraticCouplingTransform(
                        mask=mask,
                        transform_net_create_fn=create_coupling_transform_net,
                        num_bins=10,
                        tails="linear",
                    )
                )
                if i < n_layers - 1:
                    transform_layers.append(transforms.RandomPermutation(features=input_dim))

        else:
            raise ValueError(f"Unsupported flow type: {flow_type}")

        # Compose all transforms
        transform = transforms.CompositeTransform(transform_layers)

        # Create the flow
        return flows.Flow(transform, base_dist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative log probability (anomaly score) for input data.

        Args:
            x: Input tensor of shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            Negative log probability (anomaly score) for each sample
        """
        original_shape = x.shape

        # Handle 2D and 3D inputs
        if x.ndim == 2:
            # Input is already 2D, use as is
            reshaped_x = x
        elif x.ndim == 3:
            # Reshape 3D input to 2D: [batch_size * seq_len, input_dim]
            batch_size, seq_len, input_dim = x.shape
            reshaped_x = x.reshape(-1, input_dim)
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {original_shape}")

            # Convert input to float32 for flow computation
        if reshaped_x.dtype != torch.float32:
            logger.debug(f"Converting input from {reshaped_x.dtype} to float32 for flow computation")
        original_dtype = reshaped_x.dtype
        reshaped_x = reshaped_x.to(torch.float32)

        # Compute log probability
        log_prob = self.flow.log_prob(reshaped_x)

        # Return negative log probability as anomaly score
        anomaly_score = -log_prob

        # Reshape back to match input batch structure
        if x.ndim == 3:
            anomaly_score = anomaly_score.reshape(batch_size, seq_len)

        return anomaly_score

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of input data."""
        original_shape = x.shape

        if x.ndim == 2:
            reshaped_x = x
        elif x.ndim == 3:
            batch_size, seq_len, input_dim = x.shape
            reshaped_x = x.reshape(-1, input_dim)
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {original_shape}")

            # Convert input to float32 for log_prob computation
        if reshaped_x.dtype != torch.float32:
            logger.debug(f"Converting input from {reshaped_x.dtype} to float32 for log_prob computation")
        original_dtype = reshaped_x.dtype
        reshaped_x = reshaped_x.to(torch.float32)

        log_prob = self.flow.log_prob(reshaped_x)

        if x.ndim == 3:
            log_prob = log_prob.reshape(batch_size, seq_len)

        return log_prob

    def sample(self, n_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate samples from the learned distribution."""
        return self.flow.sample(n_samples, context=context)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.flow.parameters())


class NormalizingFlowFeatureModel(FeatureModel):
    """Feature model using normalizing flows for each layer."""

    def __init__(self, flows: Dict[str, NormalizingFlow]):
        super().__init__()
        # Cast to Dict[str, nn.Module] to satisfy type checker
        self.flows = utils.ModuleDict(cast(Dict[str, nn.Module], flows))

    @property
    def layer_names(self) -> list[str]:
        return list(self.flows.keys())

    def num_parameters(self) -> int:
        return sum(flow.num_parameters() for flow in self.flows.values())

    def forward(
        self, inputs: Any, features: Dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Compute anomaly scores using normalizing flows.

        Args:
            inputs: Input data (not used directly, but required by interface)
            features: Dictionary of layer activations
            return_outputs: Whether to return additional outputs

        Returns:
            Dictionary of anomaly scores (negative log probabilities) per layer
            If return_outputs=True, returns tuple of (anomaly_scores, log_probs)
        """
        anomaly_scores = {}
        log_probs = {}

        for layer_name, flow in self.flows.items():
            if layer_name not in features:
                continue

            # Compute negative log probability as anomaly score
            layer_features = features[layer_name]
            anomaly_score = flow(layer_features)

            # Store anomaly scores (these are the "losses" from the FeatureModel perspective)
            anomaly_scores[layer_name] = anomaly_score

            if return_outputs:
                log_probs[layer_name] = flow.log_prob(layer_features)

        if return_outputs:
            return anomaly_scores, log_probs

        return anomaly_scores


class NormalizingFlowDetector(FeatureModelDetector):
    """Normalizing flow-based anomaly detector."""

    def __init__(self, flows: Dict[str, NormalizingFlow], **kwargs):
        super().__init__(NormalizingFlowFeatureModel(flows), **kwargs)

    def __repr__(self):
        flow_types = [flow.flow_type for flow in self.feature_model.flows.values()]
        return f"NormalizingFlowDetector(n_flows={len(self.feature_model.flows)}, flow_types={set(flow_types)})"

    def get_flow_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about each flow for debugging/monitoring."""
        info = {}
        for layer_name, flow in self.feature_model.flows.items():
            info[layer_name] = {
                "flow_type": flow.flow_type,
                "input_dim": flow.input_dim,
                "n_layers": flow.n_layers,
                "hidden_dim": flow.hidden_dim,
                "n_blocks": flow.n_blocks,
                "dtype": flow.dtype,
            }
        return info
