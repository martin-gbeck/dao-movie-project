"""The return model: one linear regression for a film's expected log multiple, and a second fit for how widely films spread around it."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib.data_schemas import BREAK_EVEN_MULTIPLE


@dataclass
class ReturnModel:
    mean_fit: RegressionResultsWrapper
    spread_fit: GLMResultsWrapper
    # each training film's error divided by its predicted spread, sorted; the shape real errors take
    standardised_errors: np.ndarray


def fit_return_model(
    train: pl.DataFrame, formula: str, spread_formula: str
) -> ReturnModel:
    """Fit `formula` by least squares, then fit the squared errors as exp(`spread_formula`).

    The spread fit is a gamma regression with a log link, so the predicted squared error is always positive.
    """
    data = train.to_pandas()
    mean_fit = smf.ols(formula, data).fit()
    data["squared_error"] = np.asarray(mean_fit.resid) ** 2
    spread_fit = smf.glm(
        f"squared_error ~ {spread_formula}",
        data,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    standardised_errors = np.sort(
        np.asarray(mean_fit.resid) / np.sqrt(np.asarray(spread_fit.fittedvalues))
    )
    return ReturnModel(mean_fit, spread_fit, standardised_errors)


def predict_returns(
    model: ReturnModel, films: pl.DataFrame, empirical_errors: bool
) -> pl.DataFrame:
    """One row per film: the expected log multiple, its standard deviation, and the chance of a flop.

    The flop chance is the share of possible outcomes below log(BREAK_EVEN_MULTIPLE), where an outcome is the
    expected value plus the spread times a standardised error. With `empirical_errors` those errors are the
    training films' own; without, they follow a normal curve.
    """
    data = films.to_pandas()
    mean = np.asarray(model.mean_fit.predict(data))
    sd = np.sqrt(np.asarray(model.spread_fit.predict(data)))
    threshold = (np.log(BREAK_EVEN_MULTIPLE) - mean) / sd
    if empirical_errors:
        errors = model.standardised_errors
        flop_probability = np.searchsorted(errors, threshold) / len(errors)
    else:
        flop_probability = norm.cdf(threshold)
    return pl.DataFrame(
        {
            "film_id": films["film_id"],
            "predicted_log_multiple": mean,
            "predicted_sd": sd,
            "flop_probability": flop_probability,
        }
    )
