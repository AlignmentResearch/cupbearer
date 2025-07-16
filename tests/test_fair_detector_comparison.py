#!/usr/bin/env python3
"""
Fair comparison test between Normalizing Flows and VAE detectors with matched parameter counts.

This script generates synthetic data with known anomalies and compares
the performance of both detectors using architectures with similar parameter counts.
"""

import time
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from cupbearer.detectors.feature_model.nflows_detector import NormalizingFlow, NormalizingFlowDetector
from cupbearer.detectors.feature_model.vae import VAE, VAEDetector


class SyntheticDataGenerator:
    """Generate synthetic data for testing detectors."""

    def __init__(self, input_dim: int = 64, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def generate_normal_data(self, n_samples: int = 1000) -> torch.Tensor:
        """Generate normal (inlier) data from a mixture of Gaussians."""
        # Create a mixture of 3 Gaussians
        means = [
            np.random.randn(self.input_dim) * 2,
            np.random.randn(self.input_dim) * 2,
            np.random.randn(self.input_dim) * 2,
        ]

        # Generate samples from each component
        samples = []
        n_per_component = n_samples // 3

        for i, mean in enumerate(means):
            if i == len(means) - 1:  # Last component gets remaining samples
                n_comp = n_samples - len(samples)
            else:
                n_comp = n_per_component

            cov = np.eye(self.input_dim) * 0.5  # Moderate variance
            comp_samples = np.random.multivariate_normal(mean, cov, n_comp)
            samples.extend(comp_samples)

        # Shuffle samples
        samples = np.array(samples)
        np.random.shuffle(samples)

        return torch.tensor(samples, dtype=torch.float32)

    def generate_anomalous_data(self, n_samples: int = 200) -> torch.Tensor:
        """Generate anomalous (outlier) data."""
        # Mix of different anomaly types
        anomalies = []

        # Type 1: Uniform distribution (very different from Gaussian)
        n_uniform = n_samples // 3
        uniform_samples = np.random.uniform(-5, 5, (n_uniform, self.input_dim))
        anomalies.extend(uniform_samples)

        # Type 2: Single Gaussian far from normal distribution
        n_far = n_samples // 3
        far_mean = np.ones(self.input_dim) * 8  # Far from normal data
        far_cov = np.eye(self.input_dim) * 0.2  # Tight distribution
        far_samples = np.random.multivariate_normal(far_mean, far_cov, n_far)
        anomalies.extend(far_samples)

        # Type 3: High-variance Gaussian
        n_high_var = n_samples - len(anomalies)
        high_var_cov = np.eye(self.input_dim) * 5  # Very high variance
        high_var_samples = np.random.multivariate_normal(np.zeros(self.input_dim), high_var_cov, n_high_var)
        anomalies.extend(high_var_samples)

        # Shuffle anomalies
        anomalies = np.array(anomalies)
        np.random.shuffle(anomalies)

        return torch.tensor(anomalies, dtype=torch.float32)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FairDetectorComparison:
    """Compare different detectors on synthetic data using fair parameter counts."""

    def __init__(self, input_dim: int = 64, device: str = "auto"):
        self.input_dim = input_dim
        self.device = device
        self.data_generator = SyntheticDataGenerator(input_dim)

    def create_fair_vae_detector(self) -> VAEDetector:
        """Create a VAE detector with baseline parameter count."""
        # Use the baseline configuration
        vaes = {
            "layer1": VAE(
                input_dim=self.input_dim,
                latent_dim=16,
                intermediate_dim_factor=0.5,
                dtype=torch.float32,
            )
        }
        detector = VAEDetector(vaes, kld_weight=1.0)
        detector.feature_model.to(self.device)
        params = count_parameters(detector.feature_model)
        print(f"  VAE parameters: {params:,}")
        return detector

    def create_fair_nflow_detector(self, flow_type: str = "maf") -> NormalizingFlowDetector:
        """Create a normalizing flows detector with matched parameter count."""
        # Fair configurations based on parameter analysis
        fair_configs = {
            32: {
                "maf": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
                "realnvp": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
                "coupling": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
            },
            64: {
                "maf": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
                "realnvp": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
                "coupling": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
            },
            128: {
                "maf": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
                "realnvp": {"n_layers": 3, "hidden_dim": 32, "n_blocks": 1},
                "coupling": {"n_layers": 2, "hidden_dim": 32, "n_blocks": 1},
            },
        }

        config = fair_configs.get(self.input_dim, fair_configs[64])
        flow_config = config.get(flow_type, config["maf"])

        flows = {
            "layer1": NormalizingFlow(
                input_dim=self.input_dim,
                flow_type=flow_type,
                n_layers=flow_config["n_layers"],
                hidden_dim=flow_config["hidden_dim"],
                n_blocks=flow_config["n_blocks"],
                dtype=torch.float32,
                use_batch_norm=True,
                dropout_probability=0.0,
            )
        }
        detector = NormalizingFlowDetector(flows, layer_aggregation="mean")
        detector.feature_model.to(self.device)
        params = count_parameters(detector.feature_model)
        print(f"  {flow_type.upper()} parameters: {params:,}")
        return detector

    def train_detector(
        self, detector: Any, train_data: torch.Tensor, max_epochs: int = 50, batch_size: int = 64, lr: float = 1e-3
    ) -> Dict[str, Any]:
        """Train a detector and return training statistics."""
        print(f"Training {detector.__class__.__name__}...")

        # Setup training
        detector._setup_training(lr=lr)

        # Get the optimizer from the module
        optimizer = detector.module.configure_optimizers()

        # Create simple training loop
        start_time = time.time()
        losses = []

        # Convert to dataloader-like format
        n_batches = len(train_data) // batch_size

        for epoch in range(max_epochs):
            epoch_losses = []

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch_data = train_data[start_idx:end_idx]

                # Prepare batch in the format expected by detectors
                features = {"layer1": batch_data}
                inputs = Mock()  # Mock input

                # Get loss and step
                if hasattr(detector, "module"):
                    # For feature model detectors
                    loss, layer_losses = detector.module._shared_step((inputs, features))

                    # Manual optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())
                else:
                    print(f"Warning: {detector.__class__.__name__} doesn't have expected training interface")
                    break

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)

                if epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.4f}")

        training_time = time.time() - start_time

        return {
            "training_time": training_time,
            "final_loss": losses[-1] if losses else float("inf"),
            "loss_history": losses,
            "converged": len(losses) > 10 and abs(losses[-1] - losses[-5]) < 0.01,
        }

    def evaluate_detector(
        self, detector: Any, normal_data: torch.Tensor, anomalous_data: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate detector performance on test data."""
        print(f"Evaluating {detector.__class__.__name__}...")

        # Get scores for normal data
        normal_features = {"layer1": normal_data}
        normal_inputs = Mock()
        normal_layerwise_scores = detector._compute_layerwise_scores(normal_inputs, normal_features)

        # Get scores for anomalous data
        anomalous_features = {"layer1": anomalous_data}
        anomalous_inputs = Mock()
        anomalous_layerwise_scores = detector._compute_layerwise_scores(anomalous_inputs, anomalous_features)

        # Use per-sample scores instead of aggregated scores for ROC calculation
        # Extract the layer1 scores directly (these have one score per sample)
        normal_scores = normal_layerwise_scores["layer1"].flatten()
        anomalous_scores = anomalous_layerwise_scores["layer1"].flatten()

        # Combine scores and labels
        all_scores = torch.cat([normal_scores, anomalous_scores]).detach().cpu().numpy()
        all_labels = np.concatenate(
            [
                np.zeros(len(normal_scores)),  # Normal = 0
                np.ones(len(anomalous_scores)),  # Anomalous = 1
            ]
        )

        # Calculate metrics
        roc_auc = roc_auc_score(all_labels, all_scores)

        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)

        # Calculate average scores
        normal_score_mean = float(normal_scores.detach().mean())
        anomalous_score_mean = float(anomalous_scores.detach().mean())

        # Calculate separation (higher is better)
        separation = anomalous_score_mean - normal_score_mean

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "normal_score_mean": normal_score_mean,
            "anomalous_score_mean": anomalous_score_mean,
            "separation": separation,
            "normal_score_std": float(normal_scores.detach().std()),
            "anomalous_score_std": float(anomalous_scores.detach().std()),
        }

    def run_fair_comparison(
        self,
        n_train: int = 1000,
        n_test_normal: int = 200,
        n_test_anomalous: int = 200,
        max_epochs: int = 50,
        flow_types: List[str] = ["maf", "realnvp"],
    ) -> Dict[str, Any]:
        """Run fair comparison between detectors with matched parameter counts."""
        print("=== FAIR Detector Comparison Test ===")
        print(f"Input Dimension: {self.input_dim}")
        print()

        # Generate data
        print("Generating synthetic data...")
        train_data = self.data_generator.generate_normal_data(n_train)
        test_normal = self.data_generator.generate_normal_data(n_test_normal)
        test_anomalous = self.data_generator.generate_anomalous_data(n_test_anomalous)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test normal shape: {test_normal.shape}")
        print(f"Test anomalous shape: {test_anomalous.shape}")
        print()

        results = {}

        # Test VAE detector (baseline)
        print("--- VAE Detector (Baseline) ---")
        vae_detector = self.create_fair_vae_detector()
        vae_train_stats = self.train_detector(vae_detector, train_data, max_epochs=max_epochs)
        vae_eval_stats = self.evaluate_detector(vae_detector, test_normal, test_anomalous)

        results["vae"] = {
            "train_stats": vae_train_stats,
            "eval_stats": vae_eval_stats,
        }
        print()

        # Test normalizing flows detectors with fair configurations
        for flow_type in flow_types:
            print(f"--- Normalizing Flows Detector ({flow_type.upper()}) - Fair Config ---")
            nflow_detector = self.create_fair_nflow_detector(flow_type)
            nflow_train_stats = self.train_detector(nflow_detector, train_data, max_epochs=max_epochs)
            nflow_eval_stats = self.evaluate_detector(nflow_detector, test_normal, test_anomalous)

            results[f"nflow_{flow_type}"] = {
                "train_stats": nflow_train_stats,
                "eval_stats": nflow_eval_stats,
            }
            print()

        return results

    def print_fair_comparison_summary(self, results: Dict[str, Any]):
        """Print a summary of the fair comparison results."""
        print("=== FAIR COMPARISON SUMMARY ===\n")

        print("Training Performance:")
        print(f"{'Detector':<20} {'Time (s)':<10} {'Final Loss':<12} {'Converged':<10}")
        print("-" * 52)

        for detector_name, result in results.items():
            train_stats = result["train_stats"]
            print(
                f"{detector_name:<20} {train_stats['training_time']:<10.2f} "
                f"{train_stats['final_loss']:<12.4f} {train_stats['converged']:<10}"
            )

        print("\nAnomaly Detection Performance:")
        print(f"{'Detector':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'Separation':<12}")
        print("-" * 52)

        best_roc_auc = 0
        best_detector = ""

        for detector_name, result in results.items():
            eval_stats = result["eval_stats"]
            roc_auc = eval_stats["roc_auc"]
            pr_auc = eval_stats["pr_auc"]
            separation = eval_stats["separation"]

            print(f"{detector_name:<20} {roc_auc:<10.4f} {pr_auc:<10.4f} {separation:<12.4f}")

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_detector = detector_name

        print(f"\n🏆 Best detector: {best_detector} (ROC-AUC: {best_roc_auc:.4f})")

        # Compare nflows vs VAE
        if "vae" in results:
            vae_roc = results["vae"]["eval_stats"]["roc_auc"]

            nflow_results = [
                (name, result["eval_stats"]["roc_auc"]) for name, result in results.items() if name.startswith("nflow_")
            ]

            if nflow_results:
                best_nflow_name, best_nflow_roc = max(nflow_results, key=lambda x: x[1])

                print("\n📊 VAE vs Best NFlow (FAIR comparison):")
                print(f"  VAE ROC-AUC: {vae_roc:.4f}")
                print(f"  {best_nflow_name} ROC-AUC: {best_nflow_roc:.4f}")

                if best_nflow_roc > vae_roc:
                    improvement = (best_nflow_roc - vae_roc) / vae_roc * 100
                    print(f"  ✅ NFlow is better by {improvement:.1f}%")
                elif abs(best_nflow_roc - vae_roc) < 0.01:
                    print("  ⚖️  Performance is essentially equal")
                else:
                    decline = (vae_roc - best_nflow_roc) / vae_roc * 100
                    print(f"  ❌ NFlow is worse by {decline:.1f}%")

        print()


def main():
    """Main function to run the fair detector comparison."""
    print("FAIR Detector Comparison Test")
    print("=" * 50)

    # Test with different input dimensions using fair configurations
    test_configs = [
        {"input_dim": 32, "n_train": 800, "max_epochs": 40},
        {"input_dim": 64, "n_train": 1000, "max_epochs": 50},
        {"input_dim": 128, "n_train": 1200, "max_epochs": 60},
    ]

    for i, config in enumerate(test_configs):
        print(f"\n🔬 Fair Test Configuration {i + 1}: {config}")
        print("=" * 50)

        comparator = FairDetectorComparison(input_dim=config["input_dim"], device="cpu")

        results = comparator.run_fair_comparison(
            n_train=config["n_train"],
            n_test_normal=200,
            n_test_anomalous=200,
            max_epochs=config["max_epochs"],
            flow_types=["maf", "realnvp"],
        )

        comparator.print_fair_comparison_summary(results)


if __name__ == "__main__":
    main()
