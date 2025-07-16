"""Tests for normalizing flows detector."""

from unittest.mock import Mock

import pytest
import torch

from cupbearer.detectors.feature_model.nflows_detector import (
    NormalizingFlow,
    NormalizingFlowDetector,
    NormalizingFlowFeatureModel,
)


class TestNormalizingFlowDetector:
    """Test suite for normalizing flows detector."""

    def test_normalizing_flow_init(self):
        """Test NormalizingFlow initialization."""
        flow = NormalizingFlow(
            input_dim=128,
            flow_type="maf",
            n_layers=3,
            hidden_dim=256,
            n_blocks=2,
            dtype=torch.float32,
        )

        assert flow.input_dim == 128
        assert flow.flow_type == "maf"
        assert flow.n_layers == 3
        assert flow.hidden_dim == 256
        assert flow.n_blocks == 2
        assert flow.dtype == torch.float32

    def test_normalizing_flow_forward_2d(self):
        """Test NormalizingFlow forward pass with 2D input."""
        flow = NormalizingFlow(
            input_dim=64,
            flow_type="maf",
            n_layers=2,
            hidden_dim=128,
            dtype=torch.float32,
        )

        # Test 2D input
        x = torch.randn(8, 64)
        anomaly_score = flow(x)

        assert anomaly_score.shape == (8,)
        assert torch.all(torch.isfinite(anomaly_score))

    def test_normalizing_flow_forward_3d(self):
        """Test NormalizingFlow forward pass with 3D input."""
        flow = NormalizingFlow(
            input_dim=64,
            flow_type="maf",
            n_layers=2,
            hidden_dim=128,
            dtype=torch.float32,
        )

        # Test 3D input
        x = torch.randn(4, 10, 64)
        anomaly_score = flow(x)

        assert anomaly_score.shape == (4, 10)
        assert torch.all(torch.isfinite(anomaly_score))

    def test_normalizing_flow_log_prob(self):
        """Test NormalizingFlow log_prob method."""
        flow = NormalizingFlow(
            input_dim=32,
            flow_type="maf",
            n_layers=2,
            hidden_dim=64,
            dtype=torch.float32,
        )

        x = torch.randn(5, 32)
        log_prob = flow.log_prob(x)

        assert log_prob.shape == (5,)
        assert torch.all(torch.isfinite(log_prob))

        # Check that forward() returns negative log_prob
        anomaly_score = flow(x)
        assert torch.allclose(anomaly_score, -log_prob, atol=1e-6)

    def test_normalizing_flow_sample(self):
        """Test NormalizingFlow sampling."""
        flow = NormalizingFlow(
            input_dim=32,
            flow_type="maf",
            n_layers=2,
            hidden_dim=64,
            dtype=torch.float32,
        )

        samples = flow.sample(10)
        assert samples.shape == (10, 32)
        assert torch.all(torch.isfinite(samples))

    def test_different_flow_types(self):
        """Test different flow types."""
        flow_types = ["maf", "realnvp", "coupling"]

        for flow_type in flow_types:
            flow = NormalizingFlow(
                input_dim=32,
                flow_type=flow_type,
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            )

            x = torch.randn(4, 32)
            anomaly_score = flow(x)

            assert anomaly_score.shape == (4,)
            assert torch.all(torch.isfinite(anomaly_score))

    def test_normalizing_flow_feature_model(self):
        """Test NormalizingFlowFeatureModel."""
        flows = {
            "layer1": NormalizingFlow(
                input_dim=32,
                flow_type="maf",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            ),
            "layer2": NormalizingFlow(
                input_dim=32,
                flow_type="realnvp",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            ),
        }

        feature_model = NormalizingFlowFeatureModel(flows)

        assert feature_model.layer_names == ["layer1", "layer2"]

        # Test forward pass
        features = {
            "layer1": torch.randn(4, 32),
            "layer2": torch.randn(4, 32),
        }

        inputs = Mock()  # Mock input (not used directly)
        anomaly_scores = feature_model(inputs, features)

        assert set(anomaly_scores.keys()) == {"layer1", "layer2"}
        assert anomaly_scores["layer1"].shape == (4,)
        assert anomaly_scores["layer2"].shape == (4,)
        assert torch.all(torch.isfinite(anomaly_scores["layer1"]))
        assert torch.all(torch.isfinite(anomaly_scores["layer2"]))

    def test_normalizing_flow_feature_model_return_outputs(self):
        """Test NormalizingFlowFeatureModel with return_outputs=True."""
        flows = {
            "layer1": NormalizingFlow(
                input_dim=32,
                flow_type="maf",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            ),
        }

        feature_model = NormalizingFlowFeatureModel(flows)

        features = {"layer1": torch.randn(4, 32)}
        inputs = Mock()

        outputs = feature_model(inputs, features, return_outputs=True)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

        anomaly_scores, log_probs = outputs

        assert set(anomaly_scores.keys()) == {"layer1"}
        assert set(log_probs.keys()) == {"layer1"}
        assert anomaly_scores["layer1"].shape == (4,)
        assert log_probs["layer1"].shape == (4,)

    def test_normalizing_flow_detector_init(self):
        """Test NormalizingFlowDetector initialization."""
        flows = {
            "layer1": NormalizingFlow(
                input_dim=32,
                flow_type="maf",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            ),
            "layer2": NormalizingFlow(
                input_dim=32,
                flow_type="realnvp",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            ),
        }

        detector = NormalizingFlowDetector(flows, layer_aggregation="mean")

        assert isinstance(detector.feature_model, NormalizingFlowFeatureModel)
        assert detector.layer_aggregation == "mean"

        # Test repr
        repr_str = repr(detector)
        assert "NormalizingFlowDetector" in repr_str
        assert "n_flows=2" in repr_str

    def test_normalizing_flow_detector_get_flow_info(self):
        """Test NormalizingFlowDetector get_flow_info method."""
        flows = {
            "layer1": NormalizingFlow(
                input_dim=32,
                flow_type="maf",
                n_layers=3,
                hidden_dim=128,
                dtype=torch.float32,
            ),
        }

        detector = NormalizingFlowDetector(flows)
        info = detector.get_flow_info()

        assert set(info.keys()) == {"layer1"}
        assert info["layer1"]["flow_type"] == "maf"
        assert info["layer1"]["input_dim"] == 32
        assert info["layer1"]["n_layers"] == 3
        assert info["layer1"]["hidden_dim"] == 128
        assert info["layer1"]["dtype"] == torch.float32

    def test_invalid_flow_type(self):
        """Test that invalid flow type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported flow type"):
            NormalizingFlow(
                input_dim=32,
                flow_type="invalid_flow",
                n_layers=2,
                hidden_dim=64,
                dtype=torch.float32,
            )

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises ValueError."""
        flow = NormalizingFlow(
            input_dim=32,
            flow_type="maf",
            n_layers=2,
            hidden_dim=64,
            dtype=torch.float32,
        )

        # Test 1D input (should fail)
        with pytest.raises(ValueError, match="Input must be 2D or 3D"):
            flow(torch.randn(32))

        # Test 4D input (should fail)
        with pytest.raises(ValueError, match="Input must be 2D or 3D"):
            flow(torch.randn(2, 3, 4, 32))
