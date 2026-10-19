"""Scores for predictions of the log multiple and of the chance of a flop."""

from typing import cast

import polars as pl


def score_predictions(
    predictions: pl.DataFrame, films: pl.DataFrame
) -> dict[str, float]:
    """Mean absolute and root mean squared error of the log multiple, and the Brier score of the flop chance.

    `predictions` needs film_id, predicted_log_multiple and flop_probability; `films` needs film_id,
    log_multiple and flop. The Brier score is the mean of (flop chance - 1 if it flopped, else 0) squared.
    """
    scores = (
        predictions.join(films.select("film_id", "log_multiple", "flop"), on="film_id")
        .select(
            error=pl.col("predicted_log_multiple") - pl.col("log_multiple"),
            flop_miss=pl.col("flop_probability") - pl.col("flop").cast(pl.Float64),
        )
        .select(
            mae_log_multiple=pl.col("error").abs().mean(),
            rmse_log_multiple=pl.col("error").pow(2).mean().sqrt(),
            brier_flop=pl.col("flop_miss").pow(2).mean(),
        )
    )
    # one row of float means; the cast tells the type checker what polars' generic row holds
    return cast(dict[str, float], scores.row(0, named=True))
