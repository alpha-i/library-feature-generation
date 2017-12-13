import numpy as np
import logging
from scipy.special import erfinv


logging.getLogger(__name__).addHandler(logging.NullHandler())


class BinDistribution:

    def __init__(self, data, n_bins, use_centred_bins=False):

        data = data.flatten()

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
        else:
            self.bin_edges = [0]
            self.pdf = [1]
            self.bin_centres = 0
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
        """
        Computes the extend to which the variance is overestimated when using binned data.

        :return float: The amount by which the variance of the discrete pdf overestimates the continuous pdf
        """
        return np.median(self.bin_widths ** 2) / 12


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


def classify_labels(bin_edges, labels):
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
    n_bins = len(bin_edges) - 1
    binned_labels = np.zeros((n_labels, n_bins))
    nan_bins = np.array([np.nan] * n_bins)

    for i in range(n_labels):
        if np.isfinite(labels[i]):
            binned_labels[i, :], _ = np.histogram(labels[i], bin_edges, density=False)
        else:
            binned_labels[i, :] = nan_bins

    if n_label_dimensions == 2:
        binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], n_bins)
    elif n_label_dimensions == 3:
        binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], label_shape[2], n_bins)
    elif n_label_dimensions > 3:
        raise NotImplementedError("Label dimension too high:", n_label_dimensions)

    return binned_labels


def declassify_labels(dist, pdf_arrays):
    """
    Converts multiple discrete probability mass functions into means and standard deviations
    pdf_arrays has shape [n_samples, n_bins]"
    :param BinDistribution dist : The distribution generated by make_template_distribution
    :param ndarray pdf_arrays:  Of shape [n_samples, n_bins]
    :return:
    """

    point_estimates = extract_point_estimates(dist.bin_centres, pdf_arrays)

    mean = np.mean(point_estimates)
    variance = np.var(point_estimates) - dist.sheppards_correction
    variance = np.maximum(variance, dist.sheppards_correction)  # Prevent variance becoming too small

    return mean, variance


def extract_point_estimates(bin_centres, pdf_array):
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
        logging.warning('Probability mass function not normalised')
        logging.info('PDF Array shape: {}'.format(pdf_array.shape))
        logging.info('Normalisation offset: {}'.format(normalisation_offset))
        logging.info('Full pdf array: {}'.format(pdf_array))
        logging.info('Bin centres: {}'.format(bin_centres))

        logging.warning('Attempting to continue with pathological distribution')
        for i in range(n_points):
            pdf_array[i, :] = pdf_array[i, :] / np.sum(pdf_array[i, :])

    for i in range(n_points):
        pdf = pdf_array[i, :]
        points[i] = np.sum(bin_centres * pdf)

    if np.abs(normalisation_offset) > 1e-3:
        logging.info('Derived points: {}'.format(points))

    return points
