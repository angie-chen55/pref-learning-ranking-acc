import logging
import numpy as np

from abc import ABC, abstractclassmethod
from scipy import stats
from scipy.stats._stats_py import PearsonRResult
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length
from typing import Dict, List, Tuple, Union


class DistanceMetric(ABC):
    def __init__(self, eps: float = 0.001):
        self.eps = eps

    @abstractclassmethod
    def distance(
        self,
        expected: Union[List[float], List[List[float]]],
        predicted: Union[List[float], List[List[float]]],
        **kwargs,
    ) -> Union[Tuple[float, float], float]:
        pass

    def _add_eps_for_constant_array(self, arr: np.ndarray):
        # Correlation coefficients return NaN when an array has 0 variance.
        # This helper function adds eps to one element in the array (if the array
        # has 0 variance) to deal with this issue.
        if len(arr) == 0:
            return arr
        if np.all(np.equal(arr, arr[0])):
            arr[-1] += self.eps
            return arr
        return arr

    def _add_eps_for_constant_array_decorator(self, fn):
        def wrapper_fn(expected, predicted, **kwargs):
            e = self._add_eps_for_constant_array(expected)
            p = self._add_eps_for_constant_array(predicted)
            return fn(e, p, **kwargs)

        return wrapper_fn

    def _get_only_valid_values(self, expected, predicted):
        valid_idx_predicted = np.logical_and(~np.isnan(predicted), ~np.isinf(predicted))
        valid_idx_expected = np.logical_and(~np.isnan(expected), ~np.isinf(expected))
        valid_idx = np.logical_and(valid_idx_expected, valid_idx_predicted)
        return expected[valid_idx], predicted[valid_idx]


class RankingMetric(DistanceMetric):
    def __init__(
        self,
        metric_type: str = "spearman",  # choices are ['spearman', 'kendall']
    ):
        super().__init__()
        if metric_type == "spearman":
            self.metric_fn = stats.spearmanr
        elif metric_type == "kendall":
            self.metric_fn = stats.kendalltau
        else:
            raise ValueError(
                f"No metric type named {metric_type}. Possible choices: ['spearman', 'kendall]"
            )
        self.metric_fn = self._add_eps_for_constant_array_decorator(self.metric_fn)

    def distance(
        self,
        expected: Union[List[float], List[List[float]]],
        predicted: Union[List[float], List[List[float]]],
        **kwargs,
    ) -> Union[Tuple[float, List[float], List[float]], Tuple[float, float]]:
        """
        If |expected| and |predicted| are lists of lists, then we calculate the mean correlation
        for each pair of rankings. If they are non-nested lists, then we calculate a single
        correlation across the entire list.
        If |expected| and |predicted| are non-nested lists, then return (statistic, list of all p-values).
        Otherwise, return (mean statistic, list of all p-values, list of all statistics).
        """
        if len(expected) == 0 or len(predicted) == 0:
            raise ValueError(f"Expected and predicted must be non-empty.")
        assert len(expected) == len(
            predicted
        ), "Expected must be the same length as predicted."
        if isinstance(expected[0], list):
            resps = [
                self.metric_fn(e, p, **kwargs) for e, p in zip(expected, predicted)
            ]
            stat_means = np.array([r.statistic for r in resps])
            stat_mean = np.mean(stat_means[~np.isnan(stat_means)])
            pvalues = [r.pvalue for r in resps]
            return (stat_mean, pvalues, list(stat_means))
        else:
            resp = self.metric_fn(expected, predicted, **kwargs)
            stat_mean = resp.statistic
            pvalues = [resp.pvalue]
            return (stat_mean, pvalues)


class Correlation(DistanceMetric):
    def __init__(self):
        super().__init__()
        self.metric_fn = self._add_eps_for_constant_array_decorator(
            self._error_handler(self._return_nan_for_invalid_inputs(stats.pearsonr))
        )

    def _return_nan_for_invalid_inputs(self, fn):
        def wrapper_fn(expected, predicted, **kwargs):
            if np.any(np.logical_or(np.isnan(expected), np.isinf(expected))) or np.any(
                np.logical_or(np.isnan(predicted), np.isinf(predicted))
            ):
                return PearsonRResult(
                    statistic=np.nan,
                    pvalue=np.nan,
                    alternative="two-sided",
                    n=len(expected),
                    x=expected,
                    y=predicted,
                )
            return fn(expected, predicted, **kwargs)

        return wrapper_fn

    def _error_handler(self, fn):
        def wrapper_fn(expected, predicted, **kwargs):
            try:
                return fn(expected, predicted, **kwargs)
            except ValueError as err:
                if np.any(np.isnan(expected)):
                    logging.warning(f"NaNs in expected: {expected}")
                if np.any(np.isnan(predicted)):
                    logging.warning(f"NaNs in predicted: {predicted}")
                if np.any(np.isinf(expected)):
                    logging.warning(f"Infs in expected: {expected}")
                if np.any(np.isinf(predicted)):
                    logging.warning(f"Infs in predicted: {predicted}")
                logging.warning(
                    f"ValueError while computing Pearson correlation. Expected: {expected}\nPredicted: {predicted}"
                )

                raise err

        return wrapper_fn

    def distance(
        self,
        expected: Union[List[float], List[List[float]]],
        predicted: Union[List[float], List[List[float]]],
        **kwargs,
    ) -> Union[Tuple[float, List[float], List[float]], Tuple[float, float]]:
        """
        If |expected| and |predicted| are lists of lists, then we calculate the mean correlation
        for each pair of list items. If they are non-nested lists, then we calculate a single
        correlation across the entire list.
        If |expected| and |predicted| are non-nested lists, then return (statistic, list of all p-values).
        Otherwise, return (mean statistic, list of all p-values, list of all statistics).
        """
        if len(expected) == 0 or len(predicted) == 0:
            raise ValueError(f"Expected and predicted must be non-empty.")
        assert len(expected) == len(
            predicted
        ), "Expected must be the same length as predicted."
        if isinstance(expected[0], list) or isinstance(expected[0], np.ndarray):
            resps = [self.metric_fn(e, p) for e, p in zip(expected, predicted)]
            stat_means = np.array([r.statistic for r in resps])
            stat_mean = np.mean(stat_means[~np.isnan(stat_means)])
            pvalues = [r.pvalue for r in resps]
            return (stat_mean, pvalues, list(stat_means))
        else:
            resp = self.metric_fn(expected, predicted)
            stat_mean = resp.statistic
            pvalues = [resp.pvalue]
            return (stat_mean, pvalues)


class MSE(DistanceMetric):
    def distance(
        self, expected: List[float], predicted: List[float], **kwargs
    ) -> Union[Tuple[float, float], float]:
        """
        Returns (mean square error, std square error)
        """
        squared_error = [(e - p) ** 2 for e, p in zip(expected, predicted)]
        return (np.mean(squared_error), np.std(squared_error))


class Accuracy(DistanceMetric):
    def distance(
        self, expected: List[float], predicted: List[float], **kwargs
    ) -> Union[Tuple[float, float], float]:
        """
        Returns mean accuracy and also individual values.
        """
        is_correct = []
        for e, p in zip(expected, predicted):
            if (not isinstance(e, list) and np.isnan(e)) or (
                not isinstance(p, list) and np.isnan(p)
            ):
                is_correct.append(np.nan)
            else:
                is_correct.append(1 if e == p else 0)
        return np.nanmean(is_correct), is_correct


class LogDifference(DistanceMetric):
    def __init__(self, eps: float = 0.001) -> None:
        self.eps = eps

    def _replace_zeros_with_eps(self, arr: List[float]) -> List[float]:
        return [self.eps if x == 0 else x for x in arr]

    def distance(
        self, expected: List[float], predicted: List[float], **kwargs
    ) -> Union[Tuple[float, float], float]:
        """
        Returns (expected abs. log difference, std. dev. of abs. log difference)
        """
        expected = self._replace_zeros_with_eps(expected)
        predicted = self._replace_zeros_with_eps(predicted)
        abs_log_diffs = [
            np.abs(np.log(e) - np.log(p)) for e, p in zip(expected, predicted)
        ]
        return (np.mean(abs_log_diffs), np.std(abs_log_diffs))


class ExpectedCalibrationError(DistanceMetric):
    def distance(
        self,
        expected: List[float],
        predicted: List[float],
        n_bins: int = 10,
        strategy: str = "uniform",  # can be ['uniform', 'quantile']
        **kwargs,
    ) -> Union[Tuple[float, float], float]:
        """
        Returns expected calibration error.
        |expected| contains the true labels, and |predicted| contains the predicted probabilities of the positive class.
        """
        y_true = column_or_1d(expected)
        y_prob = column_or_1d(predicted)
        check_consistent_length(y_true, y_prob)

        if y_prob.min() < 0 or y_prob.max() > 1:
            raise ValueError("y_prob has values outside [0, 1].")

        labels = np.unique(y_true)
        if len(labels) > 2:
            raise ValueError(
                f"Only binary classification is supported. Provided labels {labels}."
            )
        y_true = y_true == 1

        if strategy == "quantile":  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_prob, quantiles * 100)
        elif strategy == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            raise ValueError(
                "Invalid entry to 'strategy' input. Strategy "
                "must be either 'quantile' or 'uniform'."
            )

        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        bin_sizes = bin_total[nonzero] / len(binids)

        abs_errs = [
            np.abs(p_true - p_pred) for p_true, p_pred in zip(prob_true, prob_pred)
        ]
        binned_ece = np.sum(
            [bin_size * abs_err for abs_err, bin_size in zip(abs_errs, bin_sizes)]
        )
        std_ece = np.sqrt(
            np.sum(
                [
                    bin_size * (abs_err - binned_ece) ** 2
                    for bin_size, abs_err in zip(bin_sizes, abs_errs)
                ]
            )
        )
        return (binned_ece, std_ece)


class AbsoluteDifferenceMeans(DistanceMetric):
    def distance(
        self, expected: List[float], predicted: List[float], **kwargs
    ) -> Union[Tuple[float, float], float]:
        """
        Returns |mean(expected) - mean(predicted)|
        """
        expected, predicted = self._get_only_valid_values(
            np.array(expected), np.array(predicted)
        )
        return np.abs(np.mean(expected) - np.mean(predicted))
