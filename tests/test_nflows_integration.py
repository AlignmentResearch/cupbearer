"""Integration test for normalizing flows detector with backdoor obfuscation."""

from unittest.mock import Mock, patch

import pytest
import torch

from cupbearer.detectors.feature_model.nflows_detector import NormalizingFlow, NormalizingFlowDetector


def test_nflows_detector_integration():
    """Test that NormalizingFlowDetector works within the DetectorObfuscator framework."""

    # Import the obfuscation classes
    try:
        from obfuscated_activations.train_time_experiments.src.backdoors_obfuscation import DetectorObfuscator
    except ImportError:
        pytest.skip("Obfuscation module not available")

    # Create a simple normalizing flow detector
    flows = {
        "layer1": NormalizingFlow(
            input_dim=64,
            flow_type="maf",
            n_layers=2,
            hidden_dim=128,
            dtype=torch.float32,
        ),
        "layer2": NormalizingFlow(
            input_dim=64,
            flow_type="realnvp",
            n_layers=2,
            hidden_dim=128,
            dtype=torch.float32,
        ),
    }

    detector = NormalizingFlowDetector(flows, layer_aggregation="mean")

    # Create the obfuscator
    obfuscator = DetectorObfuscator(
        detector=detector,
        detector_lr=1e-4,
        detector_ignore_steps=5,  # Small for testing
        device="cpu",
    )

    # Mock data structures
    batch_size = 4
    seq_len = 10

    # Mock model outputs with hidden states
    def create_mock_output(hidden_dim=64):
        mock_output = Mock()
        mock_output.logits = torch.randn(batch_size, seq_len, 32000)  # vocab size
        mock_output.hidden_states = [
            torch.randn(batch_size, seq_len, hidden_dim),  # layer 0
            torch.randn(batch_size, seq_len, hidden_dim),  # layer 1
            torch.randn(batch_size, seq_len, hidden_dim),  # layer 2
        ]
        return mock_output

    backdoor_output = create_mock_output()
    normal_output = create_mock_output()

    # Mock tokens and masks
    backdoor_tokens = torch.randint(0, 32000, (batch_size, seq_len))
    normal_tokens = torch.randint(0, 32000, (batch_size, seq_len))

    # Create simple masks (all True for simplicity)
    backdoor_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    normal_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Test data structures
    backdoor_data = (backdoor_output, backdoor_tokens, None, backdoor_mask)
    normal_data = (normal_output, normal_tokens, None, normal_mask)

    activation_matching_layers = [1, 2]

    # Test the obfuscation loss computation
    # First few steps should return 0 (detector ignore steps)
    for step in range(3):
        loss, layer_losses = obfuscator.compute_obfuscation_loss(
            backdoor_data, normal_data, activation_matching_layers
        )
        assert loss.item() == 0.0
        assert isinstance(layer_losses, dict)

    # After ignore steps, should return actual loss
    for step in range(3):
        loss, layer_losses = obfuscator.compute_obfuscation_loss(
            backdoor_data, normal_data, activation_matching_layers
        )
        assert loss.item() != 0.0
        assert torch.isfinite(loss)
        assert isinstance(layer_losses, dict)

        # Check that layer losses are logged
        assert len(layer_losses) > 0
        for key, value in layer_losses.items():
            assert "layer_losses/NormalizingFlowDetector" in key
            assert torch.isfinite(torch.tensor(value))


def test_construct_nflow_obfuscation_loss_fn():
    """Test the constructor function for normalizing flows obfuscation loss."""

    try:
        from obfuscated_activations.train_time_experiments.src.backdoors_obfuscation import (
            construct_nflow_obfuscation_loss_fn,
        )
    except ImportError:
        pytest.skip("Obfuscation module not available")

    # Mock tokenizer
    tokenizer = Mock()

    # Create the obfuscation loss function
    layers = [1, 2, 3]
    obfuscation_loss_fn = construct_nflow_obfuscation_loss_fn(
        layers=layers,
        tokenizer=tokenizer,
        obfuscate_over="full_prompt",
        flow_type="maf",
        n_layers=2,
        hidden_dim=64,
        dtype=torch.float32,
        detector_ignore_steps=5,
    )

    # Check that the function was created
    assert callable(obfuscation_loss_fn)
    assert hasattr(obfuscation_loss_fn, "__name__")
    assert "NormalizingFlowDetector" in obfuscation_loss_fn.__name__

    # Mock the compute_mask function to return simple masks
    with patch("obfuscated_activations.train_time_experiments.src.backdoors_obfuscation.compute_mask") as mock_compute_mask:
        mock_compute_mask.return_value = torch.ones(4, 10, dtype=torch.bool)

        # Mock data
        batch_size = 4
        seq_len = 10
        hidden_dim = 4096  # Standard transformer size

        def create_mock_output():
            mock_output = Mock()
            mock_output.logits = torch.randn(batch_size, seq_len, 32000)
            mock_output.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_dim),  # layer 0
                torch.randn(batch_size, seq_len, hidden_dim),  # layer 1
                torch.randn(batch_size, seq_len, hidden_dim),  # layer 2
                torch.randn(batch_size, seq_len, hidden_dim),  # layer 3
            ]
            return mock_output

        backdoor_output = create_mock_output()
        normal_output = create_mock_output()

        backdoor_tokens = torch.randint(0, 32000, (batch_size, seq_len))
        normal_tokens = torch.randint(0, 32000, (batch_size, seq_len))

        backdoor_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        normal_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        backdoor_data = (backdoor_output, backdoor_tokens, backdoor_mask, backdoor_mask)
        normal_data = (normal_output, normal_tokens, normal_mask, normal_mask)

        activation_matching_layers = layers

        # Test the obfuscation loss function
        # First few calls should return 0 (detector ignore steps)
        for _ in range(3):
            result = obfuscation_loss_fn(backdoor_data, normal_data, activation_matching_layers)
            if isinstance(result, tuple):
                loss, layer_losses = result
                assert loss.item() == 0.0
                assert isinstance(layer_losses, dict)
            else:
                assert result.item() == 0.0

        # After ignore steps, should return actual loss
        for _ in range(3):
            result = obfuscation_loss_fn(backdoor_data, normal_data, activation_matching_layers)
            if isinstance(result, tuple):
                loss, layer_losses = result
                assert loss.item() != 0.0
                assert torch.isfinite(loss)
                assert isinstance(layer_losses, dict)
            else:
                assert result.item() != 0.0
                assert torch.isfinite(result)


def test_nflows_detector_training_compatibility():
    """Test that NormalizingFlowDetector is compatible with the training framework."""

    # Create a simple detector
    flows = {
        "layer1": NormalizingFlow(
            input_dim=32,
            flow_type="maf",
            n_layers=2,
            hidden_dim=64,
            dtype=torch.float32,
        ),
    }

    detector = NormalizingFlowDetector(flows, layer_aggregation="mean")

    # Test _setup_training method
    detector._setup_training(lr=1e-4)

    # Check that the module was created
    assert hasattr(detector, "module")
    assert detector.module is not None

    # Test _compute_layerwise_scores method
    features = {"layer1": torch.randn(4, 32)}
    inputs = Mock()

    scores = detector._compute_layerwise_scores(inputs, features)

    assert isinstance(scores, dict)
    assert "layer1" in scores
    assert scores["layer1"].shape == (4,)
    assert torch.all(torch.isfinite(scores["layer1"]))


if __name__ == "__main__":
    # Run a simple test to verify the implementation works
    print("Testing normalizing flows detector...")
    test_nflows_detector_integration()
    test_construct_nflow_obfuscation_loss_fn()
    test_nflows_detector_training_compatibility()
    print("All tests passed!")
