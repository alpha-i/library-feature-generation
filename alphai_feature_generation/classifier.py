import logging

import numpy as np
from scipy.special import erfinv

logger = logging.getLogger(__name__)


class BinDistribution:

    def __init__(self, data, n_bins, use_centred_bins=False):

        data = data.flatten()
        data = data[np.isfinite(data)]

        self.n_bins = n_bins
        self.pdf_type = self.find_best_fit_pdf_type()

        self.mean = np.mean(data)
        self.median = np.median(data)
        self.sigma = np.std(data)

        n_datapoints = len(data)
        if n_datapoints > 0:
            self.bin_edges = self._compute_balanced_bin_edges(data, use_centred_bins)
            counts, bins = np.histogram(data, self.bin_edges, density=False)

            if use_centred_bins:  # Catch outliers
                counts[0] += np.sum(data < self.bin_edges[0])
                counts[-1] += np.sum(data > self.bin_edges[-1])

            self.pdf = counts / n_datapoints
            self.n_bins = n_bins
            self.bin_centres = self._compute_bin_centres()
            self.bin_widths = self._compute_bin_widths()
            self.mean_bin_width = self._calc_mean_bin_width()
            self.sheppards_correction = self._calc_sheppards_correction()
            self.weighted_bin_centres = self._compute_weighted_bin_centres(data)
        else:
            self.bin_edges = [0]
            self.pdf = [1]
            self.bin_centres = 0
            self.weighted_bin_centres = 0
            self.bin_widths = 0
            self.mean_bin_width = 0
            self.sheppards_correction = 0

    def find_best_fit_pdf_type(self):
        """
        Returns the best-fit functional form for the given data. Currently

        :return: String: Type of pdf which best describes the data
        """
        return 'Gaussian'  # TODO: enable tests for t-distributed data; lognormal, etc

    def _compute_balanced_bin_edges(self, data, use_centred_bins):
        """
        Finds the bins needed such that they equally divided the data.

        :param bool centred_bins: whether to force the bins to be centred on zero
        :return ndarray: of length (n_bins + 1) which defines edges of bins which contain an equal number of data points
        """

        if data.ndim != 1:
            raise ValueError("Currently only supports one dimensional input")
        n_xvals = len(data)
        n_array = np.arange(n_xvals)

        if use_centred_bins:
            unit_gaussian_edges = _calculate_unit_gaussian_edges(self.n_bins + 1)
            bin_edges = unit_gaussian_edges * self.sigma
        else:  # Original bin definition
            xrange = np.linspace(0, n_xvals - 1, self.n_bins + 1)
            bin_edges = np.interp(xrange, n_array, np.sort(data))

        return bin_edges

    def _compute_bin_centres(self):
        """
        Finds the bin centres

        :return: ndarray The corresponding centres of the bins
        """
        return 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

    def _compute_weighted_bin_centres(self, data):
        """
        Finds the bin centres

        :return: ndarray The corresponding centres of the bins
        """

        weighted_bin_centres = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            lo_edge = self.bin_edges[i]
            hi_edge = self.bin_edges[i+1]

            single_bin_data = data[(data >= lo_edge) & (data <= hi_edge)]
            weighted_estimate = np.mean(single_bin_data)

            # Could yield nans if bins are empty.
            if np.isnan(weighted_estimate):
                weighted_bin_centres[i] = self.bin_centres[i]
            else:
                weighted_bin_centres[i] = weighted_estimate

        return weighted_bin_centres

    def _compute_bin_widths(self):
        """
        Finds the bin widths

        :return ndarray: The corresponding edges of the bins
        """
        return self.bin_edges[1:] - self.bin_edges[:-1]

    def _calc_mean_bin_width(self):
        """
        Finds the average bin width given a set of edges

        :return float:  The average bin width
        """

        n_bins = len(self.bin_edges) - 1
        full_gap = np.abs(self.bin_edges[-1] - self.bin_edges[0])

        return full_gap / n_bins

    def _calc_sheppards_correction(self):
        """ Computes the extend to which the variance is overestimated when using binned data.

        :return float: The amount by which the variance of the discrete pdf overestimates the continuous pdf
        """
        return np.median(self.bin_widths ** 2) / 12

    def estimate_confidence_interval(self, pdf, confidence_interval=0.68):
        """ Returns median and outer ranges of confidence interval. Default is 68%"""

        median_val = 0.5
        hi_val = median_val + confidence_interval / 2
        low_val = median_val - confidence_interval / 2

        median = self._calculate_single_confidence_interval(pdf, median_val)
        lower_bound = self._calculate_single_confidence_interval(pdf, low_val)
        upper_bound = self._calculate_single_confidence_interval(pdf, hi_val)

        return median, lower_bound, upper_bound

    def _calculate_single_confidence_interval(self, pdf, confidence_interval):
        """ Estimate confidence level of a histogram by stepping through bins until we hit desired threshold. """
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError("Invalid confidence interval {} requested.".format(confidence_interval))

        pdf = pdf.flatten()
        if not len(pdf) == self.n_bins:
            raise ValueError("Pdf {} of length {} doesnt match expected number of bins {}".format(pdf, len(pdf), self.n_bins))

        # First verify length of input
        bin_index = 0
        cumulative_sum = pdf[bin_index]

        while cumulative_sum < confidence_interval:
            bin_index += 1
            cumulative_sum += pdf[bin_index]

        # Now we know bin_index holds the median value. Just need to interpolate a bit
        bin_total = pdf[bin_index]

        if bin_index == 0:  # Treat edges of histogram as triangles
            bin_offset = self.bin_edges[1] - self.weighted_bin_centres[0]
            triangle_width = 3 * bin_offset  # Triangle CoM is 1/3
            low_edge = self.bin_edges[1] - triangle_width
            area_fill_fraction = confidence_interval / bin_total
            linear_fill_fraction = np.sqrt(area_fill_fraction)
            value = low_edge + triangle_width * linear_fill_fraction
        elif bin_index == self.n_bins:
            bin_offset = self.weighted_bin_centres[-1] - self.bin_edges[-2]
            triangle_width = 3 * bin_offset  # Triangle CoM is 1/3
            upper_edge = self.bin_edges[-2] + triangle_width
            area_fill_fraction = (1 - confidence_interval) / bin_total
            linear_fill_fraction = np.sqrt(area_fill_fraction)
            value = upper_edge - triangle_width * linear_fill_fraction
        else:
            lower_edge = self.bin_edges[bin_index]
            bin_width = self.bin_widths[bin_index]

            overflow = cumulative_sum - confidence_interval
            residue = bin_total - overflow

            bin_fraction = residue / bin_total

            value = lower_edge + bin_width * bin_fraction

        return value

    def declassify_single_pdf(self, pdf_array, use_median=True):
        """  Here we keep the multple pdfs seperate, yielding a mean and variance for each.

        :param pdf_array:
        :param use_median:
        :return:
        """
        point_estimates = self.extract_point_estimates(pdf_array, use_median)

        mean = np.mean(point_estimates)
        variance = np.sum(self.weighted_bin_centres ** 2 * pdf_array) - mean ** 2
        variance -= self.sheppards_correction
        variance = np.maximum(variance, self.sheppards_correction)  # Prevent variance becoming too small

        return point_estimates, variance

    def extract_point_estimates(self, pdf_array, use_median):
        """
        Finds the mean values of discrete probability mass functions into point estimates, taken to be the mean
        bin_centres has shape [n_bins]
        :param ndarray bin_centres: One-dimensional array
        :param ndarray pdf_array: One or two-dimensional array corresponding to probability mass functions
        :return ndarray: means of the pdfs
        """

        if pdf_array.ndim == 1:
            pdf_array = np.expand_dims(pdf_array, axis=0)

        n_points = pdf_array.shape[0]
        points = np.zeros(n_points)

        normalisation_offset = np.sum(pdf_array[0, :]) - 1.0

        if np.abs(normalisation_offset) > 1e-3:
            logger.warning('Probability mass function not normalised')
            logger.debug('PDF Array shape: {}'.format(pdf_array.shape))
            logger.debug('Normalisation offset: {}'.format(normalisation_offset))
            logger.debug('Full pdf array: {}'.format(pdf_array))
            logger.debug('Bin centres: {}'.format(self.bin_centres))

            logger.debug('Attempting to continue with pathological distribution')
            for i in range(n_points):
                pdf_array[i, :] = pdf_array[i, :] / np.sum(pdf_array[i, :])

        for i in range(n_points):
            pdf = pdf_array[i, :]
            if use_median:
                points[i] = self.calculate_discrete_median(pdf)
            else:
                points[i] = np.sum(self.bin_centres * pdf)

        if np.abs(normalisation_offset) > 1e-3:
            logger.debug('Derived points: {}'.format(points))

        return points

    def classify_labels(self, labels):
        """
        Takes numerical values and returns their binned values in one-hot format
        :param ndarray bin_edges: One dimensional array
        :param ndarray labels:  One or two dimensional array (e.g. could be [batch_size, n_series])
        :return ndarray: dimensions [labels.shape, n_bins]
        """

        n_label_dimensions = labels.ndim
        label_shape = labels.shape
        labels = labels.flatten()

        n_labels = len(labels)
        binned_labels = np.zeros((n_labels, self.n_bins))
        nan_bins = np.array([np.nan] * self.n_bins)

        for i in range(n_labels):
            if np.isfinite(labels[i]):
                binned_labels[i, :], _ = np.histogram(labels[i], self.bin_edges, density=False)
            else:
                binned_labels[i, :] = nan_bins

        if n_label_dimensions == 2:
            binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], self.n_bins)
        elif n_label_dimensions == 3:
            binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], label_shape[2], self.n_bins)
        elif n_label_dimensions > 3:
            raise ValueError("Label dimension too high:", n_label_dimensions)

        return binned_labels


def _calculate_unit_gaussian_edges(n_edges):
    """ Retrieve array of edges for a unit gaussian such that the bins hold equal probability

    :param nparray n_bins:
    :return: nparray edges: The bin edges
    """

    stepsize = 2 / n_edges
    startval = -1 + stepsize / 2
    stopval = 1
    sampler = np.arange(startval, stopval, stepsize)

    gaussian_edges = erfinv(sampler) * np.sqrt(2)

    return gaussian_edges


