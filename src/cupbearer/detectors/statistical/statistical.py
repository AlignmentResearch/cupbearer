from abc import abstractmethod

import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from cupbearer.detectors.activation_based import ActivationBasedDetector
from cupbearer.detectors.statistical.helpers import update_covariance


class StatisticalDetector(ActivationBasedDetector):
    use_trusted: bool = True
    use_untrusted: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def init_variables(self, sample_batch, case: str):
        pass

    @abstractmethod
    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        pass

    def _finalize_training(self, **kwargs):
        pass

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        *,
        pbar: bool = True,
        max_steps: int | None = None,
        device: torch.device | str = "auto",
        **kwargs,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        all_dataloaders = {}

        if self.use_trusted:
            if trusted_dataloader is None:
                raise ValueError(f"{self.__class__.__name__} requires trusted training data.")
            all_dataloaders["trusted"] = trusted_dataloader
        if self.use_untrusted:
            if untrusted_dataloader is None:
                raise ValueError(f"{self.__class__.__name__} requires untrusted training data.")
            all_dataloaders["untrusted"] = untrusted_dataloader

        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            for case, dataloader in all_dataloaders.items():
                logger.debug(f"Collecting statistics on {case} data")

                self.init_variables(
                    sample_batch=next(iter(dataloader)),
                    case=case,
                    device=device,
                )

                if pbar:
                    dataloader = tqdm(dataloader, total=max_steps or len(dataloader))

                for i, (_, activations) in enumerate(dataloader):
                    activations = {k: v.to(device) for k, v in activations.items()}
                    if max_steps and i >= max_steps:
                        break
                    self.batch_update(activations, case)

        self._finalize_training(**kwargs)


class ActivationCovarianceBasedDetector(StatisticalDetector):
    """Generic abstract detector that learns means and covariance matrices
    during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._means = {}
        self._Cs = {}
        self._ns = {}

    def init_variables(
        self,
        sample_batch,
        case: str,
        device: torch.device | str | None = None,
    ):
        _, example_activations = sample_batch

        # v is an entire batch, v[0] are activations for a single input
        activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
        if device is None:
            device = next(iter(example_activations.values())).device

        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only learn "
                "covariances along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        logger.debug("Activation sizes: \n" + "\n".join(f"{k}: {size}" for k, size in activation_sizes.items()))
        self._means[case] = {k: torch.zeros(size[-1], device=device) for k, size in activation_sizes.items()}
        self._Cs[case] = {k: torch.zeros((size[-1], size[-1]), device=device) for k, size in activation_sizes.items()}
        self._ns[case] = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            activation = rearrange(activation, "batch ... dim -> (batch ...) dim")
            assert activation.ndim == 2, activation.shape
            (
                self._means[case][k],
                self._Cs[case][k],
                self._ns[case][k],
            ) = update_covariance(
                self._means[case][k],
                self._Cs[case][k],
                self._ns[case][k],
                activation,
            )

    @abstractmethod
    def post_covariance_training(self, **kwargs):
        pass

    @abstractmethod
    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        """Compute the anomaly score for a single layer.

        `name` is passed in to access the mean/covariance and any custom derived
        quantities computed in post_covariance_training.

        `activation` can be a batch (batch, dim) or a batch with sequence (batch, seq_len, dim).
        It returns a score for each sample in the batch and the sequence dimension.

        Should return a tensor of shape (batch,) or (batch, seq_len) with the anomaly scores.
        """
        pass

    def _compute_layerwise_scores(self, inputs, features):
        batch_shape = next(iter(features.values())).shape
        batch_size = batch_shape[0]
        features = {k: rearrange(v, "batch ... dim -> (batch ...) dim") for k, v in features.items()}
        scores = {k: self._individual_layerwise_score(k, v) for k, v in features.items()}
        scores = {
            k: rearrange(
                v,
                "(batch independent_dims) -> batch independent_dims",
                batch=batch_size,
            )
            for k, v in scores.items()
        }
        if len(batch_shape) == 2:
            scores = {k: v.squeeze(1) for k, v in scores.items()}
        return scores

    def _finalize_training(self, **kwargs):
        # Post process.
        # We're using no_grad() instead of inference_mode() since in some cases,
        # we later want gradients for anomaly scores.
        with torch.no_grad():
            self.means = self._means
            self.covariances = {}
            for case, Cs in self._Cs.items():
                self.covariances[case] = {k: C / (self._ns[case][k] - 1) for k, C in Cs.items()}
                if any(torch.count_nonzero(C) == 0 for C in self.covariances[case].values()):
                    logger.warning(
                        f"Found zero covariance matrix in {case} data. "
                        "There may be a problem with the data or feature extractor."
                    )

            self.post_covariance_training(**kwargs)

    def num_parameters(self) -> int:
        return sum(v.numel() for v in self._means.values()) + sum(v.numel() for v in self._Cs.values())

    def to(self, device: torch.device | str):
        super().to(device)
        self._means = {case: {k: v.to(device) for k, v in means.items()} for case, means in self._means.items()}
        self._Cs = {case: {k: v.to(device) for k, v in Cs.items()} for case, Cs in self._Cs.items()}
