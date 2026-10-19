# Movie studio data

Every file in `raw/` is exactly as downloaded. All cleaning and joining happens in `movie_studio.py` (section 1.2, "Parse and join") and `lib/parsing.py`.

## Files the notebook uses

Ranked by how much the analysis leans on each one.

| File | Rows | Used for | Source |
|---|---|---|---|
| `raw/movie_data.csv` | 3,647 films, 1927-2016 | The base table: genres, runtime, rating, country, director, lead actors | DAO2702X course file |
| `raw/the_numbers/movie-budgets.csv` | 5,460 films | Release date, budget in USD, worldwide gross, story source, production method | The Numbers, via the delafields scrape |
| `raw/tmdb/tmdb_5000_movies.csv` | 4,803 films, 1916-2017 | Dates, budgets and grosses The Numbers lacks; production companies give the studio | TMDB 5000 on Kaggle |
| `raw/fred/CPIAUCSL.csv` | monthly, Jan 1947 to Sep 2026 | Puts every budget in 2016 dollars | FRED |

## Files not used yet

| File | Rows | What it holds | Source |
|---|---|---|---|
| `raw/lock5/HollywoodMovies_2012_2018.csv` | 1,295 films, 2012-2018 | Lead studio, Rotten Tomatoes and audience scores, opening weekend, budget and grosses in $M | Lock5Data, `HollywoodMovies2018` |
| `raw/lock5/HollywoodMovies_2e.csv` | 970 films, 2007-2013 | The same, plus a `Story` plot type (Quest, Metamorphosis...) | Lock5Data, `HollywoodMovies` (2nd edition) |
| `raw/tmdb/tmdb_5000_credits.csv` | 4,803 films | Full cast and crew as JSON, keyed on `movie_id` | TMDB 5000 on Kaggle |
| `raw/the_numbers_json/` | 56 files of about 100 films | The scrape before it was merged into `movie-budgets.csv` | delafields scrape |

## Sources

1. **`movie_data.csv`** came with the DAO2702X course (NUS). Its columns (`facenumber_in_poster`, the Facebook likes, `imdb_score`) match the Kaggle IMDB 5000 Movie Dataset, scraped from IMDb in 2016, so it is most likely a cleaned subset of that: <https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset>. The course added `gross_inf_adj`, `Inf_factor` and `budget_inf_adj`. Not confirmed with the course.
2. **The Numbers** budget table, <https://www.the-numbers.com/movie/budgets/all>, scraped by GitHub user delafields on 5 Dec 2019: <https://github.com/delafields/movie-budgets>, write-up at <https://delafields.github.io/movie-budgets/>. `the_numbers_json/README.md` is that repo's own README and lists every column.
3. **TMDB 5000**, The Movie Database's export of about 5,000 films, published on Kaggle in 2017: <https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>. Our copy came from a mirror of that set on Hugging Face.
4. **FRED CPIAUCSL**, the Consumer Price Index for All Urban Consumers, all items, seasonally adjusted, 1982-84 = 100. Published by the U.S. Bureau of Labor Statistics and served by the St. Louis Fed: <https://fred.stlouisfed.org/series/CPIAUCSL>. The copy here runs to Sep 2026.
5. **Lock5** datasets from Lock, Lock, Lock Morgan, Lock and Lock, *Statistics: Unlocking the Power of Data*, shipped in the R package Lock5Data (<https://cran.r-project.org/package=Lock5Data>) and as CSVs on the book's site, for example <https://www.lock5stat.com/datasets3e/HollywoodMovies.csv>. Column docs: <https://r-packages.io/datasets/HollywoodMovies> and <https://r-packages.io/datasets/HollywoodMovies2018>.

## Units and traps, file by file

### `movie_data.csv`

- Money is in millions of dollars. Interstellar shows `budget` 165 and `gross` 187.99.
- `gross` is the US box office only. Worldwide gross comes from The Numbers or TMDB.
- Budgets for non-US films are in their own currency: *The Host* shows 12,215, which is millions of won. Take the dollar budget from The Numbers or TMDB.
- The inflation columns are in 2016 dollars: `gross_inf_adj = gross / Inf_factor`, with `Inf_factor` at 1.0 for 2016. The notebook builds its own adjustment from FRED instead of using these.
- Facebook likes are in thousands (Christopher Nolan shows 22).
- The file starts with a byte-order mark and has some text saved in the wrong encoding (`PenÃ©lope`), so read it with `encoding="utf8-lossy"` and decode the names again.
- 42 films appear twice with the same title, year and director.

### `the_numbers/movie-budgets.csv`

- Money and dates are text: `"$400,000,000"`, `"Apr 23, 2019"`, and runtime is `"181 minutes"`.
- In about 300 rows the scrape shifted columns, so `story_source` or `production_method` holds text like "All Time Domestic Box Office for...".
- The first column has no name; it is the pandas index from the scrape.

### `tmdb/tmdb_5000_movies.csv`

- A `budget` or `revenue` of 0 means unknown, not zero. Both are in plain dollars, not millions.
- `genres` and `production_companies` are JSON strings.

### `lock5/`

- Money is in millions of dollars.
- `Profitability` is world gross as a percentage of budget: 345.3 means the film made 3.45 times its budget.

### `the_numbers_json/`

- The `.json` files are gzipped despite the extension.
- `create_csv.py` will not run as it is: it imports `urls_to_scrape`, which is not in the folder, and reads from `./data/`.

### All sources

Coverage leans toward films people remember. The Numbers keeps roughly the 5,000 biggest budgets, and every source thins out before 1980, so old and small films that flopped are mostly missing.
