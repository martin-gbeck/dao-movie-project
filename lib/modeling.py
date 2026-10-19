"""The return model: one linear regression for a film's log worldwide gross, with the errors of its budget tier as its range."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib.data_schemas import BREAK_EVEN_MULTIPLE


@dataclass
class ReturnModel:
    fit: RegressionResultsWrapper
    # the training films' errors (actual minus predicted log gross) in each budget tier, sorted
    tier_errors: dict[str, np.ndarray]


def fit_return_model(train: pl.DataFrame, formula: str) -> ReturnModel:
    """Fit `formula` by ordinary least squares on `train` and keep the training errors of each budget tier.

    `train` needs every column the formula names, plus budget_tier.
    """
    fit = smf.ols(formula, train.to_pandas()).fit()
    errors = train.select(pl.col("budget_tier").cast(pl.String)).with_columns(
        error=pl.Series(np.asarray(fit.resid))
    )
    tier_errors = {
        str(tier): np.sort(group["error"].to_numpy())
        for (tier,), group in errors.group_by("budget_tier")
    }
    return ReturnModel(fit, tier_errors)


def predict_returns(model: ReturnModel, films: pl.DataFrame) -> pl.DataFrame:
    """One row per film: the predicted log worldwide gross, the spread of its tier's errors, and the chance of a flop.

    `films` needs film_id, budget_tier, budget_real_musd and every column the formula names. A film's possible
    outcomes are its prediction plus each training error of its budget tier; the flop chance is the share of
    those outcomes below BREAK_EVEN_MULTIPLE times its budget.
    """
    predicted = np.asarray(model.fit.predict(films.to_pandas()))
    tiers = films["budget_tier"].cast(pl.String).to_list()
    break_even = np.log(BREAK_EVEN_MULTIPLE * films["budget_real_musd"].to_numpy())
    flop_probability = np.array(
        [
            np.searchsorted(model.tier_errors[tier], gap) / len(model.tier_errors[tier])
            for tier, gap in zip(tiers, break_even - predicted, strict=True)
        ]
    )
    return pl.DataFrame(
        {
            "film_id": films["film_id"],
            "predicted_log_gross": predicted,
            "predicted_sd": [float(model.tier_errors[tier].std()) for tier in tiers],
            "flop_probability": flop_probability,
        }
    )
