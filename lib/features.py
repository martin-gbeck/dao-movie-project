"""Features a studio knows before release: the competition around a release date and the talent's track record."""

import polars as pl


def count_competitors(
    films: pl.DataFrame, calendar: pl.DataFrame, window_days: int
) -> pl.DataFrame:
    """Count the other releases within `window_days` before or after each film's release date.

    `films` needs film_id, title_key, release_date and budget_usd; `calendar` needs title_key, release_date
    and budget_usd for every known release. Returns one row per film: competitors, big_competitors (budget at
    least as large as the film's own) and competitor_budget_musd (their budgets added up, in USD millions).
    A film with nobody in its window gets zeros.
    """
    others = calendar.select(
        pl.col("title_key").alias("other_key"),
        pl.col("release_date").alias("other_date"),
        pl.col("budget_usd").alias("other_budget"),
    )
    window = pl.duration(days=window_days)
    counts = (
        films.select(
            "film_id",
            "title_key",
            "budget_usd",
            window_start=pl.col("release_date") - window,
            window_end=pl.col("release_date") + window,
        )
        .join_where(
            others,
            pl.col("other_date") >= pl.col("window_start"),
            pl.col("other_date") <= pl.col("window_end"),
            pl.col("other_key") != pl.col("title_key"),
        )
        .group_by("film_id")
        .agg(
            pl.len().alias("competitors"),
            (pl.col("other_budget") >= pl.col("budget_usd"))
            .sum()
            .alias("big_competitors"),
            (pl.col("other_budget").sum() / 1e6).alias("competitor_budget_musd"),
        )
    )
    return (
        films.select("film_id")
        .join(counts, on="film_id", how="left")
        .with_columns(pl.exclude("film_id").fill_null(0))
    )


def prior_track_record(
    films: pl.DataFrame, person_column: str, prefix: str
) -> pl.DataFrame:
    """Summarise each person's earlier films, using only films released before the one in the row.

    `films` needs film_id, release_date, log_multiple, flop and `person_column` (a director or actor name).
    Returns one row per film with `<prefix>_prior_films`, `<prefix>_prior_median_log_multiple` and
    `<prefix>_prior_flop_share`; the last two are null for a first film in the data.
    """
    history = films.select(
        pl.col(person_column),
        pl.col("release_date").alias("earlier_date"),
        pl.col("log_multiple").alias("earlier_log_multiple"),
        pl.col("flop").alias("earlier_flop"),
    ).drop_nulls(person_column)
    record = (
        films.select("film_id", "release_date", person_column)
        .join(history, on=person_column, how="inner")
        .filter(pl.col("earlier_date") < pl.col("release_date"))
        .group_by("film_id")
        .agg(
            pl.len().alias(f"{prefix}_prior_films"),
            pl.col("earlier_log_multiple")
            .median()
            .alias(f"{prefix}_prior_median_log_multiple"),
            pl.col("earlier_flop").mean().alias(f"{prefix}_prior_flop_share"),
        )
    )
    return (
        films.select("film_id")
        .join(record, on="film_id", how="left")
        .with_columns(pl.col(f"{prefix}_prior_films").fill_null(0))
    )
