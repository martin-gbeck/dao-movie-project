"""The data model for the joined film table: one row per film, money in USD."""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig

# worldwide gross per dollar of production budget a film needs to break even (assumption, see the objective)
BREAK_EVEN_MULTIPLE = 2.5


class Film(pa.DataFrameModel):
    # fmt: off
    film_id: int = pa.Field(ge=0, unique=True)                  # row id after duplicates are removed
    title: str                                                  # title as in the course dataset, encoding fixed
    title_key: str                                              # normalised title used to match sources
    release_date: pl.Date                                       # first US release (The Numbers, else TMDB)
    year: int = pa.Field(ge=1915, le=2017)                      # release year
    budget_usd: float = pa.Field(gt=0)                          # production budget, nominal USD
    budget_real_musd: float = pa.Field(gt=0)                    # production budget, USD millions at 2016 prices
    worldwide_gross_usd: float = pa.Field(gt=0)                 # worldwide box office, nominal USD
    worldwide_gross_real_musd: float = pa.Field(gt=0)           # worldwide box office, USD millions at 2016 prices
    domestic_gross_usd: float = pa.Field(ge=0, nullable=True)   # US and Canada box office, nominal USD
    gross_multiple: float = pa.Field(gt=0)                      # worldwide gross / budget
    log_multiple: float                                         # natural log of gross_multiple
    flop: bool                                                  # gross_multiple below BREAK_EVEN_MULTIPLE
    genres: str                                                 # IMDb genres joined by "|", a film has several
    content_rating: str = pa.Field(nullable=True)               # MPAA rating such as PG-13 or R
    duration_min: int = pa.Field(gt=0)                          # runtime in minutes
    country: str                                                # main production country
    story_source: str = pa.Field(nullable=True)                 # original screenplay, book, comic, real events...
    production_method: str = pa.Field(nullable=True)            # live action, animation...
    studio: str                                                 # major parent studio among the producers, or Independent
    director: str = pa.Field(nullable=True)                     # director name
    actor_1: str = pa.Field(nullable=True)                      # first-billed actor
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
