"""The data model for the joined film table: one row per film, money in USD."""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig

# worldwide gross per dollar of production budget a film needs to break even (assumption, see the objective)
BREAK_EVEN_MULTIPLE = 2.5


class Film(pa.DataFrameModel):
    # fmt: off
    film_id: int = pa.Field(ge=0, unique=True)                     # row id
    imdb_id: str = pa.Field(unique=True, str_startswith="tt")      # IMDb title id, the key to IMDb and Wikidata
    title: str                                                     # title
    release_date: pl.Date                                          # US release (The Numbers), else TMDB's date
    year: int = pa.Field(ge=1980, le=2024)                         # release year
    budget_usd: float = pa.Field(gt=0)                             # production budget, nominal USD
    budget_real_musd: float = pa.Field(gt=0)                       # production budget, USD millions at 2024 prices
    worldwide_gross_usd: float = pa.Field(ge=10_000)               # worldwide box office, nominal USD
    worldwide_gross_real_musd: float = pa.Field(gt=0)              # worldwide box office, USD millions at 2024 prices
    domestic_gross_usd: float = pa.Field(ge=0, nullable=True)      # US and Canada box office, nominal USD (The Numbers only)
    log_gross: float                                               # natural log of worldwide_gross_real_musd: the target
    gross_multiple: float = pa.Field(gt=0)                         # worldwide gross / budget
    flop: bool                                                     # gross_multiple below BREAK_EVEN_MULTIPLE
    genres: str = pa.Field(nullable=True)                          # IMDb genres joined by ",", up to three
    runtime_min: int = pa.Field(gt=0, nullable=True)               # IMDb runtime in minutes
    english: bool                                                  # original language is English
    us_production: bool = pa.Field(nullable=True)                  # the United States is among the production countries
    rating: str                                                    # MPAA rating: G, PG, PG-13, R, or Other (NC-17, not rated, unknown)
    story_source: str = pa.Field(nullable=True)                    # The Numbers' story source: original screenplay, book, comic...
    based_on_work: bool                                            # Wikidata or TMDB keywords say it adapts another work
    series: str = pa.Field(nullable=True)                          # Wikidata film series, if any
    sequel: bool                                                   # a later film in its series, or tagged as a sequel
    major: str                                                     # major parent studio among producers and distributors, or Independent
    money_source: str = pa.Field(isin=["The Numbers", "TMDB"])    # where budget and gross were taken from
    # fmt: on

    class Config(BaseConfig):  # ty: ignore[invalid-mutable-override]
        coerce = True
        strict = "filter"

    @pa.dataframe_check
    def multiple_is_gross_over_budget(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            (pl.col("worldwide_gross_usd") / pl.col("budget_usd")).is_close(
                pl.col("gross_multiple"), abs_tol=1e-9
            )
        )

    @pa.dataframe_check
    def flop_follows_break_even(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col("flop") == (pl.col("gross_multiple") < BREAK_EVEN_MULTIPLE)
        )
