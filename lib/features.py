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


def career_before_release(
    people: pl.DataFrame, filmography: pl.DataFrame
) -> pl.DataFrame:
    """What each person on each film had done in earlier years, in the same role.

    `people` has one row per film and person: film_id, nconst, role, film_year. `filmography` has one row per
    credit: nconst, role, credit_year, imdb_rating, imdb_votes, log_gross (null for films without box office
    in the film table) and flop. Only credits from years before `film_year` count, so everything here was
    public when the film was greenlit. Returns per film and person: prior_films (all earlier feature films),
    prior_rating (mean IMDb rating of earlier films with at least 1,000 votes), prior_box_office_films,
    prior_median_log_gross and prior_flop_share (over the earlier films with known box office).
    """
    rated = pl.col("imdb_votes") >= 1_000
    return (
        people.join(filmography, on=["nconst", "role"], how="left")
        .filter(pl.col("credit_year") < pl.col("film_year"))
        .group_by("film_id", "nconst", "role")
        .agg(
            pl.len().alias("prior_films"),
            pl.col("imdb_rating").filter(rated).mean().alias("prior_rating"),
            pl.col("log_gross").count().alias("prior_box_office_films"),
            pl.col("log_gross").median().alias("prior_median_log_gross"),
            pl.col("flop").mean().alias("prior_flop_share"),
        )
        .join(
            people.select("film_id", "nconst", "role"),
            on=["film_id", "nconst", "role"],
            how="right",
        )
        .with_columns(pl.col("prior_films", "prior_box_office_films").fill_null(0))
    )
