"""Scores for predictions of the log worldwide gross and of the chance of a flop."""

from typing import cast

import polars as pl


def score_predictions(
    predictions: pl.DataFrame, films: pl.DataFrame
) -> dict[str, float]:
    """Mean absolute and root mean squared error of the log gross, R², and the Brier score of the flop chance.

    `predictions` needs film_id, predicted_log_gross and flop_probability; `films` needs film_id, log_gross and
    flop. R² is 1 - (squared error / squared distance from the films' own mean). The Brier score is the mean of
    (flop chance - 1 if it flopped, else 0) squared.
    """
    scores = (
        predictions.join(films.select("film_id", "log_gross", "flop"), on="film_id")
        .select(
            error=pl.col("predicted_log_gross") - pl.col("log_gross"),
            spread=pl.col("log_gross") - pl.col("log_gross").mean(),
            flop_miss=pl.col("flop_probability") - pl.col("flop").cast(pl.Float64),
        )
        .select(
            mae_log_gross=pl.col("error").abs().mean(),
            rmse_log_gross=pl.col("error").pow(2).mean().sqrt(),
            r_squared=1 - pl.col("error").pow(2).sum() / pl.col("spread").pow(2).sum(),
            brier_flop=pl.col("flop_miss").pow(2).mean(),
        )
    )
    # one row of float means; the cast tells the type checker what polars' generic row holds
    return cast(dict[str, float], scores.row(0, named=True))
