import numpy as np
import unittest

from alphai_feature_generation.classifier import BinDistribution

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

    def test_classify_labels(self):
        distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        labels = np.array([0.5, 9.5])  # should be assigned to first and last bins
        classes = distribution.classify_labels(labels)

        class1 = np.zeros(N_BINS)
        class2 = np.zeros(N_BINS)
        class1[0] = 1
        class2[-1] = 1

        correct_classes = [class1, class2]
        self.assertTrue(np.allclose(classes, correct_classes, rtol=RTOL, atol=ATOL))

    # def test_calculate_single_confidence_interval(self):
    #     distribution = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
    #
    #     median_confidence_interval = 0.5
    #     high_confidence_interval = 0.975
    #     TEST_PDF_1 = distribution.pdf
    #     value_1 = distribution._calculate_single_confidence_interval(TEST_PDF_1, median_confidence_interval)
    #     value_2 = distribution._calculate_single_confidence_interval(TEST_PDF_1, high_confidence_interval)
    #
    #     TEST_PDF_2 = np.zeros(N_BINS)
    #     TEST_PDF_2[-1] = 1
    #
    #     value_3 = distribution._calculate_single_confidence_interval(TEST_PDF_2, median_confidence_interval)
    #     value_4 = distribution._calculate_single_confidence_interval(TEST_PDF_2, high_confidence_interval)
    #
    #     upper_bound = TEST_EDGES[-1]
    #     true_value_1 = upper_bound / 2  # Median of the 0-10 range
    #     true_value_2 = upper_bound * 0.975  # 97.5% confidence for U(0-10)
    #     true_value_3 = upper_bound * 0.95  # Median of the 0.9-1 bin
    #     true_value_4 = upper_bound * 0.9975  # 97.5% confidence for that bin
    #
    #     self.assertAlmostEqual(value_1, true_value_1, delta=RTOL)
    #     self.assertAlmostEqual(value_2, true_value_2, delta=RTOL)
    #     self.assertAlmostEqual(value_3, true_value_3, delta=RTOL)
    #     self.assertAlmostEqual(value_4, true_value_4, delta=RTOL)

    def test_declassify_labels(self):
        # Check the mean and variance of a simple pdf [00001000]
        test_classification = np.zeros(N_BINS)
        test_classification[5] = 1
        bin_width = 1

        dist = BinDistribution(TEST_TRAIN_LABELS, N_BINS)
        pdf_arrays = dist.pdf

        mean, variance = dist.declassify_labels(pdf_arrays)
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

        distribution = BinDistribution(TEST_BIN_CENTRES, N_BINS)
        estimated_points = distribution.extract_point_estimates(pdf_array, False)
        point_a = TEST_BIN_CENTRES[index_a]
        point_b = TEST_BIN_CENTRES[index_b]
        points = [point_a, point_b]

        self.assertTrue(np.allclose(estimated_points, points, rtol=RTOL, atol=ATOL))

