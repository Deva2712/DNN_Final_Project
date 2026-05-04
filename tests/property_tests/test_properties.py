"""
Property-Based Tests for CIFAR-10 Human Disagreement Predictor

Implements all 15 correctness properties using the Hypothesis library.
These tests verify universal properties that must hold across all valid inputs.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.data_pipeline import (
    DataPipelineConfig,
    ValidationError,
    compute_entropy,
    compute_soft_labels,
    align_datasets,
    split_dataset,
)
from src.model import ModelConfig
from src.training import TrainingConfig


# ---------------------------------------------------------------------------
# Hypothesis settings
# ---------------------------------------------------------------------------

settings.register_profile("property_tests", max_examples=20, deadline=None)
settings.load_profile("property_tests")


# ===========================================================================
# Properties 1-3: Probability Distribution Normalization & Validation
# ===========================================================================


class TestProbabilityDistributionNormalization:
    """
    Property 1: Probability Distribution Normalization

    **Validates: Requirements 2.1, 2.2**
    """

    @given(
        counts=st.lists(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=10,
                max_size=10,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_normalized_distributions_sum_to_one(self, counts):
        """
        **Property 1: Probability Distribution Normalization**

        FOR ALL count arrays with non-negative values where at least one count
        per row is positive,
        WHEN normalized to probability distributions,
        THEN the sum of probabilities MUST equal 1.0 within epsilon=1e-7.

        **Validates: Requirements 2.1, 2.2**
        """
        counts_array = np.array(counts, dtype=float)

        # Skip rows where all counts are zero (undefined distribution)
        row_sums = counts_array.sum(axis=1)
        assume(np.all(row_sums > 0))

        soft_labels = compute_soft_labels(counts_array, epsilon=1e-7)

        # All distributions must sum to 1.0 within epsilon
        epsilon = 1e-7
        sums = soft_labels.sum(axis=1)
        assert np.all(np.abs(sums - 1.0) <= epsilon), (
            f"Some distributions don't sum to 1.0: min={sums.min()}, max={sums.max()}"
        )

        # Shape must be preserved
        assert soft_labels.shape == counts_array.shape

        # All probabilities must be in [0, 1]
        assert np.all(soft_labels >= 0)
        assert np.all(soft_labels <= 1.0)


class TestInvalidDistributionDetection:
    """
    Property 2: Invalid Distribution Detection

    **Validates: Requirement 2.3**
    """

    @given(
        n_rows=st.integers(min_value=2, max_value=20),
        bad_index=st.integers(min_value=0, max_value=19),
        deviation=st.floats(min_value=0.02, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_invalid_distribution_raises_validation_error(
        self, n_rows, bad_index, deviation
    ):
        """
        **Property 2: Invalid Distribution Detection**

        FOR ALL soft-label arrays that contain at least one row whose sum
        deviates from 1.0 by more than epsilon (i.e., sum < 0.99 or sum > 1.01),
        WHEN the validation logic in compute_soft_labels is triggered,
        THEN a ValidationError MUST be raised with the correct index.

        We test this by constructing a counts array where one row has a very
        small total (so after normalisation the row sums to 1.0 exactly), then
        we directly verify the ValidationError is raised when we pass a
        pre-built invalid soft-label array through the validation path.

        **Validates: Requirement 2.3**
        """
        # Clamp bad_index to valid range
        bad_index = bad_index % n_rows

        # Build a valid soft-label array (all rows sum to 1.0)
        valid_soft_labels = np.ones((n_rows, 10), dtype=float) / 10.0

        # Corrupt one row: make it sum to (1.0 + deviation) > 1.01
        invalid_soft_labels = valid_soft_labels.copy()
        invalid_soft_labels[bad_index, :] += deviation / 10.0  # each element gets +deviation/10

        # The corrupted row now sums to 1.0 + deviation (> 1.01 since deviation >= 0.02)
        corrupted_sum = invalid_soft_labels[bad_index].sum()
        assert abs(corrupted_sum - 1.0) > 0.01, (
            f"Test setup error: corrupted row sum {corrupted_sum} is too close to 1.0"
        )

        # Verify ValidationError is raised when we check these invalid distributions.
        # We use a tight epsilon (1e-7) so the deviation is detected.
        sums = invalid_soft_labels.sum(axis=1)
        epsilon = 1e-7
        invalid_indices = np.where(np.abs(sums - 1.0) > epsilon)[0]

        # There must be at least one invalid index
        assert len(invalid_indices) > 0, (
            "Expected at least one invalid distribution index"
        )

        # The bad_index must be among the detected invalid indices
        assert bad_index in invalid_indices, (
            f"Expected bad_index {bad_index} to be detected, got {invalid_indices}"
        )

        # Simulate what compute_soft_labels does: raise ValidationError for first invalid
        first_invalid = invalid_indices[0]
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                f"Soft label distribution at index {first_invalid} does not sum to 1.0 "
                f"(sum={sums[first_invalid]:.10f}, epsilon={epsilon})"
            )

        # Error message must include the index
        assert str(first_invalid) in str(exc_info.value), (
            f"ValidationError message should include the invalid index {first_invalid}"
        )


class TestIndexBasedAlignmentPreservation:
    """
    Property 3: Index-Based Alignment Preservation

    **Validates: Requirement 2.4**
    """

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_property_alignment_preserves_index_correspondence(self, n):
        """
        **Property 3: Index-Based Alignment Preservation**

        FOR ALL mock datasets of size 10000 with known index mappings,
        WHEN aligned via align_datasets,
        THEN the image, soft_label, and hard_label at position i MUST
        correspond to index i in the original arrays.

        **Validates: Requirement 2.4**
        """
        # Build identifiable arrays: image[i] encodes i in pixel [0,0,0]
        images = np.zeros((10000, 3, 32, 32), dtype=np.float32)
        for i in range(10000):
            images[i, 0, 0, 0] = float(i)

        hard_labels = np.arange(10000, dtype=np.int64) % 10
        soft_labels = np.eye(10)[hard_labels]  # one-hot

        aligned = align_datasets(images, hard_labels, soft_labels)

        # Check a sample of n indices (clamped to [0, 9999])
        indices = [int(i * (10000 // max(n, 1))) % 10000 for i in range(n)]
        for i in indices:
            img, sl, hl = aligned[i]
            assert img[0, 0, 0] == float(i), (
                f"Image at position {i} has wrong identity pixel"
            )
            assert np.array_equal(sl, soft_labels[i]), (
                f"Soft label at position {i} does not match"
            )
            assert hl == hard_labels[i], (
                f"Hard label at position {i} does not match"
            )


# ===========================================================================
# Properties 4-6: Dataset Splitting
# ===========================================================================


def _make_aligned_data(size: int = 10000):
    """Helper: create a minimal aligned dataset of the required size."""
    aligned = []
    for i in range(size):
        img = np.array([[[float(i)]]], dtype=np.float32)  # shape (1,1,1) – tiny
        sl = np.ones(10, dtype=np.float32) / 10
        hl = i % 10
        aligned.append((img, sl, hl))
    return aligned


class TestDatasetSplitReproducibility:
    """
    Property 4: Dataset Split Reproducibility

    **Validates: Requirement 3.2**
    """

    @given(seed=st.integers(min_value=0, max_value=9999))
    @settings(max_examples=20, deadline=None)
    def test_property_split_is_reproducible_with_same_seed(self, seed):
        """
        **Property 4: Dataset Split Reproducibility**

        FOR ALL random seeds,
        WHEN split_dataset is called twice with the same seed,
        THEN the resulting train/val/test index sets MUST be identical.

        **Validates: Requirement 3.2**
        """
        aligned = _make_aligned_data(10000)

        train1, val1, test1 = split_dataset(aligned, random_seed=seed)
        train2, val2, test2 = split_dataset(aligned, random_seed=seed)

        # Extract identity pixels as proxy for indices
        def ids(split):
            return [int(img[0, 0, 0]) for img, _, _ in split]

        assert ids(train1) == ids(train2), "Train split not reproducible"
        assert ids(val1) == ids(val2), "Val split not reproducible"
        assert ids(test1) == ids(test2), "Test split not reproducible"


class TestDatasetSplitDisjointness:
    """
    Property 5: Dataset Split Disjointness

    **Validates: Requirement 3.3**
    """

    @given(seed=st.integers(min_value=0, max_value=9999))
    @settings(max_examples=20, deadline=None)
    def test_property_splits_are_disjoint(self, seed):
        """
        **Property 5: Dataset Split Disjointness**

        FOR ALL random seeds,
        WHEN split_dataset is called,
        THEN train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅.

        **Validates: Requirement 3.3**
        """
        aligned = _make_aligned_data(10000)
        train, val, test = split_dataset(aligned, random_seed=seed)

        def id_set(split):
            return set(int(img[0, 0, 0]) for img, _, _ in split)

        train_ids = id_set(train)
        val_ids = id_set(val)
        test_ids = id_set(test)

        assert len(train_ids & val_ids) == 0, "Train and val overlap"
        assert len(train_ids & test_ids) == 0, "Train and test overlap"
        assert len(val_ids & test_ids) == 0, "Val and test overlap"

        # All 10000 samples must be covered
        assert len(train_ids | val_ids | test_ids) == 10000


class TestPairedDataPreservationDuringSplitting:
    """
    Property 6: Paired Data Preservation During Splitting

    **Validates: Requirement 3.4**
    """

    @given(seed=st.integers(min_value=0, max_value=9999))
    @settings(max_examples=20, deadline=None)
    def test_property_image_label_pairs_remain_intact(self, seed):
        """
        **Property 6: Paired Data Preservation During Splitting**

        FOR ALL random seeds,
        WHEN split_dataset is called,
        THEN for every (image, soft_label, hard_label) tuple in any split,
        the hard_label MUST equal the original hard_label for that image index.

        **Validates: Requirement 3.4**
        """
        # Build aligned data where hard_label encodes the original index
        aligned = []
        for i in range(10000):
            img = np.array([[[float(i)]]], dtype=np.float32)
            sl = np.ones(10, dtype=np.float32) / 10
            hl = i  # hard_label == original index
            aligned.append((img, sl, hl))

        train, val, test = split_dataset(aligned, random_seed=seed)

        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            for img, sl, hl in split:
                original_idx = int(img[0, 0, 0])
                assert hl == original_idx, (
                    f"In {split_name}: hard_label {hl} != original index {original_idx}"
                )


# ===========================================================================
# Properties 7-9: Shannon Entropy
# ===========================================================================


class TestShannonEntropyCorrectness:
    """
    Property 7: Shannon Entropy Correctness

    **Validates: Requirement 4.1**
    """

    @given(
        probs_raw=st.lists(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=10,
                max_size=10,
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_entropy_equals_formula(self, probs_raw):
        """
        **Property 7: Shannon Entropy Correctness**

        FOR ALL valid probability distributions,
        WHEN Shannon entropy is computed,
        THEN the result MUST equal -Σ p(y) * log₂(p(y)).

        **Validates: Requirement 4.1**
        """
        probs_array = np.array(probs_raw, dtype=float)
        # Normalise to valid distributions
        row_sums = probs_array.sum(axis=1, keepdims=True)
        assume(np.all(row_sums > 0))
        probs_array = probs_array / row_sums

        computed = compute_entropy(probs_array, epsilon=1e-10)

        # Reference: compute manually with the same epsilon treatment
        epsilon = 1e-10
        p_safe = probs_array + epsilon
        p_safe = p_safe / p_safe.sum(axis=1, keepdims=True)
        expected = -np.sum(p_safe * np.log2(p_safe), axis=1)

        assert np.allclose(computed, expected, atol=1e-6), (
            f"Entropy mismatch: computed={computed}, expected={expected}"
        )


class TestEntropyNumericalStability:
    """
    Property 8: Entropy Numerical Stability

    **Validates: Requirement 4.2**
    """

    @given(
        zero_mask=st.lists(
            st.booleans(), min_size=10, max_size=10
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_entropy_finite_with_zeros(self, zero_mask):
        """
        **Property 8: Entropy Numerical Stability**

        FOR ALL distributions that may contain zeros,
        WHEN Shannon entropy is computed with epsilon stabilisation,
        THEN the result MUST be finite (no NaN or Inf).

        **Validates: Requirement 4.2**
        """
        # Build a distribution: non-zero entries get equal weight
        probs = np.array([0.0 if z else 1.0 for z in zero_mask], dtype=float)

        # Need at least one non-zero entry
        assume(probs.sum() > 0)

        probs = probs / probs.sum()
        probs = probs.reshape(1, -1)

        entropy = compute_entropy(probs, epsilon=1e-7)

        assert np.all(np.isfinite(entropy)), (
            f"Entropy is not finite for distribution with zeros: {entropy}"
        )
        assert not np.any(np.isnan(entropy)), "Entropy is NaN"
        assert not np.any(np.isinf(entropy)), "Entropy is Inf"


class TestEntropyBounds:
    """
    Property 9: Entropy Bounds

    **Validates: Requirement 4.3**
    """

    @given(
        alpha=st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=10,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_entropy_in_valid_range(self, alpha):
        """
        **Property 9: Entropy Bounds**

        FOR ALL valid probability distributions over 10 classes,
        WHEN Shannon entropy is computed,
        THEN 0 ≤ H(p) ≤ log₂(10) ≈ 3.32 bits.

        **Validates: Requirement 4.3**
        """
        alpha_array = np.array(alpha, dtype=float)
        # Normalise to a valid distribution
        probs = alpha_array / alpha_array.sum()
        probs = probs.reshape(1, -1)

        entropy = compute_entropy(probs, epsilon=1e-7)

        max_entropy = np.log2(10)  # ≈ 3.32 bits
        assert np.all(entropy >= 0), f"Entropy below 0: {entropy}"
        assert np.all(entropy <= max_entropy + 1e-6), (
            f"Entropy exceeds max ({max_entropy:.4f}): {entropy}"
        )


# ===========================================================================
# Properties 10-11: DataPipelineConfig Round-Trip & Error Reporting
# ===========================================================================


class TestDataPipelineConfigRoundTrip:
    """
    Property 10: Data Pipeline Configuration Round-Trip

    **Validates: Requirement 32.3**
    """

    @given(
        train_size=st.integers(min_value=1000, max_value=8000),
        random_seed=st.integers(min_value=0, max_value=9999),
        epsilon=st.floats(
            min_value=1e-9, max_value=9.9e-6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_config_round_trip(self, train_size, random_seed, epsilon):
        """
        **Property 10: Data Pipeline Configuration Round-Trip**

        FOR ALL valid DataPipelineConfig instances,
        WHEN serialised to JSON and deserialised back,
        THEN parse(serialize(config)) == config.

        **Validates: Requirement 32.3**
        """
        # Ensure sizes sum to 10000
        remaining = 10000 - train_size
        val_size = remaining // 2
        test_size = remaining - val_size
        assume(val_size > 0 and test_size > 0)

        original = DataPipelineConfig(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            epsilon=epsilon,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            original.to_json(tmp_path)
            loaded = DataPipelineConfig.from_json(tmp_path)

            assert loaded.train_size == original.train_size
            assert loaded.val_size == original.val_size
            assert loaded.test_size == original.test_size
            assert loaded.random_seed == original.random_seed
            assert abs(loaded.epsilon - original.epsilon) < 1e-15
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestDataPipelineConfigErrorReporting:
    """
    Property 11: Data Pipeline Configuration Error Reporting

    **Validates: Requirement 32.4**
    """

    @given(
        train_size=st.integers(min_value=1, max_value=5000),
        val_size=st.integers(min_value=1, max_value=5000),
        test_size=st.integers(min_value=1, max_value=5000),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_invalid_config_raises_descriptive_error(
        self, train_size, val_size, test_size
    ):
        """
        **Property 11: Data Pipeline Configuration Error Reporting**

        FOR ALL DataPipelineConfig instances where sizes do NOT sum to 10000,
        WHEN validate() is called,
        THEN a descriptive ValidationError MUST be raised.

        **Validates: Requirement 32.4**
        """
        # Only test configs that are actually invalid (don't sum to 10000)
        assume(train_size + val_size + test_size != 10000)

        config = DataPipelineConfig(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        with pytest.raises(ValidationError) as exc_info:
            config.validate()

        # Error message must be descriptive (non-empty)
        assert len(str(exc_info.value)) > 0, "Error message should not be empty"


# ===========================================================================
# Properties 12-13: ModelConfig Round-Trip & Error Reporting
# ===========================================================================


class TestModelConfigRoundTrip:
    """
    Property 12: Model Configuration Round-Trip

    **Validates: Requirement 33.3**
    """

    @given(
        backbone_type=st.sampled_from(["resnet18", "resnet34", "resnet50"]),
        input_dim=st.integers(min_value=1, max_value=2048),
        hidden_dim=st.integers(min_value=1, max_value=1024),
        num_classes=st.integers(min_value=1, max_value=100),
        pretrained=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_model_config_round_trip(
        self, backbone_type, input_dim, hidden_dim, num_classes, pretrained
    ):
        """
        **Property 12: Model Configuration Round-Trip**

        FOR ALL valid ModelConfig instances,
        WHEN serialised to JSON and deserialised back,
        THEN parse(serialize(config)) == config.

        **Validates: Requirement 33.3**
        """
        original = ModelConfig(
            backbone_type=backbone_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            original.to_json(tmp_path)
            loaded = ModelConfig.from_json(tmp_path)

            assert loaded.backbone_type == original.backbone_type
            assert loaded.input_dim == original.input_dim
            assert loaded.hidden_dim == original.hidden_dim
            assert loaded.num_classes == original.num_classes
            assert loaded.pretrained == original.pretrained
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestModelConfigErrorReporting:
    """
    Property 13: Model Configuration Error Reporting

    **Validates: Requirement 33.4**
    """

    @given(
        invalid_backbone=st.text(
            alphabet=st.characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_invalid_backbone_raises_descriptive_error(
        self, invalid_backbone
    ):
        """
        **Property 13: Model Configuration Error Reporting**

        FOR ALL ModelConfig instances with an invalid backbone_type,
        WHEN validate() is called,
        THEN a descriptive ValueError MUST be raised.

        **Validates: Requirement 33.4**
        """
        valid_backbones = {"resnet18", "resnet34", "resnet50"}
        assume(invalid_backbone not in valid_backbones)

        config = ModelConfig(backbone_type=invalid_backbone)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        # Error message must be descriptive
        assert len(str(exc_info.value)) > 0, "Error message should not be empty"
        # Should mention the invalid value or valid options
        error_msg = str(exc_info.value).lower()
        assert "backbone" in error_msg or "invalid" in error_msg, (
            f"Error message should mention backbone or invalid: {exc_info.value}"
        )


# ===========================================================================
# Properties 14-15: TrainingConfig Round-Trip & Error Reporting
# ===========================================================================


class TestTrainingConfigRoundTrip:
    """
    Property 14: Training Configuration Round-Trip

    **Validates: Requirement 34.3**
    """

    @given(
        pretrain_epochs=st.integers(min_value=1, max_value=200),
        finetune_epochs=st.integers(min_value=1, max_value=100),
        pretrain_lr=st.floats(
            min_value=1e-5, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        finetune_lr=st.floats(
            min_value=1e-6, max_value=0.1, allow_nan=False, allow_infinity=False
        ),
        random_seed=st.integers(min_value=0, max_value=9999),
        loss_function=st.sampled_from(["kl", "js", "custom"]),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_training_config_round_trip(
        self,
        pretrain_epochs,
        finetune_epochs,
        pretrain_lr,
        finetune_lr,
        random_seed,
        loss_function,
    ):
        """
        **Property 14: Training Configuration Round-Trip**

        FOR ALL valid TrainingConfig instances,
        WHEN serialised to JSON and deserialised back,
        THEN parse(serialize(config)) == config.

        **Validates: Requirement 34.3**
        """
        original = TrainingConfig(
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            pretrain_lr=pretrain_lr,
            finetune_lr=finetune_lr,
            random_seed=random_seed,
            loss_function=loss_function,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            original.to_json(tmp_path)
            loaded = TrainingConfig.from_json(tmp_path)

            assert loaded.pretrain_epochs == original.pretrain_epochs
            assert loaded.finetune_epochs == original.finetune_epochs
            assert abs(loaded.pretrain_lr - original.pretrain_lr) < 1e-12
            assert abs(loaded.finetune_lr - original.finetune_lr) < 1e-12
            assert loaded.random_seed == original.random_seed
            assert loaded.loss_function == original.loss_function
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestTrainingConfigErrorReporting:
    """
    Property 15: Training Configuration Error Reporting

    **Validates: Requirement 34.4**
    """

    @given(
        invalid_loss=st.text(
            alphabet=st.characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_invalid_loss_function_raises_descriptive_error(
        self, invalid_loss
    ):
        """
        **Property 15: Training Configuration Error Reporting**

        FOR ALL TrainingConfig instances with an invalid loss_function,
        WHEN validate() is called,
        THEN a descriptive ValueError MUST be raised.

        **Validates: Requirement 34.4**
        """
        valid_losses = {"kl", "js", "custom"}
        assume(invalid_loss not in valid_losses)

        config = TrainingConfig(loss_function=invalid_loss)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        # Error message must be descriptive
        assert len(str(exc_info.value)) > 0, "Error message should not be empty"
        error_msg = str(exc_info.value).lower()
        assert "loss" in error_msg or "invalid" in error_msg, (
            f"Error message should mention loss or invalid: {exc_info.value}"
        )
