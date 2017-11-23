import numpy as np
import unittest

from alphai_feature_generation.classifier import (
    BinDistribution,
    classify_labels,
    declassify_labels,
    extract_point_estimates
)

from tests.helpers import (
    N_BINS,
    MIN_EDGE,
    MAX_EDGE,
    TEST_EDGES,
    TEST_BIN_CENTRES,
    TEST_TRAIN_LABELS,
    RTOL,
    ATOL,
)


class TestBinDistribution(unittest.TestCase):

    def test_template_distribution_construct(self):
        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)

        self.assertIsInstance(distribution, BinDistribution)

    def test_compute_bin_centres(self):
        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        true_centre = 1.5
        self.assertTrue(np.allclose(distribution.bin_centres[1], true_centre, rtol=RTOL, atol=ATOL))

    def test_compute_bin_widths(self):

        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        true_widths = np.ones(shape=N_BINS)

        self.assertTrue(np.allclose(distribution.bin_widths, true_widths, rtol=RTOL, atol=ATOL))

    def test_compute_balanced_bin_edges(self):
        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        self.assertTrue(np.allclose(distribution.bin_edges, TEST_EDGES, rtol=RTOL, atol=ATOL))

    def test_calc_mean_bin_width(self):
        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        true_mean = (MAX_EDGE - MIN_EDGE) / N_BINS
        self.assertAlmostEqual(distribution.mean_bin_width, true_mean)

    def test_calc_sheppards_correction(self):

        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        correction = np.mean(distribution.bin_widths) ** 2 / 12

        self.assertAlmostEqual(distribution.sheppards_correction, correction)


class TestClassifier(unittest.TestCase):

    def test_classify_labels(self):
        true_classification = np.zeros(N_BINS)
        true_classification[5] = 1
        label = np.array([5.01])

        binned_label = classify_labels(TEST_EDGES, label)
        self.assertTrue(np.allclose(binned_label, true_classification, rtol=RTOL, atol=ATOL))

    def test_declassify_labels(self):
        # Check the mean and variance of a simple pdf [00001000]
        test_classification = np.zeros(N_BINS)
        test_classification[5] = 1
        bin_width = 1

        dist = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        pdf_arrays = dist.pdf

        mean, variance = declassify_labels(dist, pdf_arrays)
        true_mean = np.mean(TEST_TRAIN_LABELS)

        true_variance = bin_width ** 2 / 12

        self.assertAlmostEqual(mean, true_mean)
        self.assertAlmostEqual(variance, true_variance)

    def test_extract_point_estimates(self):
        # Set up a mock of two pdfs
        pdf_array = np.zeros(shape=(2, N_BINS))
        index_a = 2
        index_b = 5
        pdf_array[0, index_a] = 1
        pdf_array[1, index_b] = 1

        estimated_points = extract_point_estimates(TEST_BIN_CENTRES, pdf_array)
        point_a = TEST_BIN_CENTRES[index_a]
        point_b = TEST_BIN_CENTRES[index_b]
        points = [point_a, point_b]

        self.assertTrue(np.allclose(estimated_points, points, rtol=RTOL, atol=ATOL))
