# What should a film studio greenlight?

DAO2702X (Programming for Business Analytics, NUS) group project, team Jupyter.

A studio decides before filming how much to spend, on what kind of film, with which director and cast, and when to release it. We combine nine data sources into one table of 8,355 films released from 1980 to 2024, and use it to show which of those choices raise a film's box office, which raise or lower the chance of a flop, and what slate of films a studio should build for the risk it can carry.

## The brief

From Appendix C of the DAO2702 course overview: one practical business problem on real data.

| Part | What | Weight | When |
|---|---|---|---|
| Report | 8-page PDF with no code, plus the notebook as HTML, zipped as `Project_Jupyter` | 30% | Fri 13 Nov 23:59 |
| Presentation | 15 minutes for the whole team, everyone speaks, no scripts, graded individually | 30% | Weeks 11-12 (26 Oct to 6 Nov), slot not booked yet |

The report is graded on data visualization 50%, problem statement 20% (novelty counts), analysis 25% and writing 5%. The presentation comes before the report, so the charts and findings have to exist first.

## The question and the targets

Which choices a studio makes before filming change what a film takes in, and how often it loses money?

| Target | Definition | Meaning |
|---|---|---|
| `log_gross` | natural log of worldwide gross, USD millions at 2024 prices | what the model predicts; +1 means 2.7 times the box office |
| `gross_multiple` | worldwide gross / production budget | 3.0 means the film took in 3 dollars at the box office for every dollar it cost to make |
| `flop` | `gross_multiple` < 2.5 | the film did not earn back its costs |

The 2.5 line is an assumption. The studio keeps about half of the box office, and marketing adds roughly a quarter of the production budget again, so a film that costs 100 costs 125 in total and needs 250 at the box office to cover it. Trade rules of thumb run from 2 to 3 times the budget, and the main results get checked at both ends.

## Decided

- Only what is known before the studio commits to the film goes into the model (19 Oct). Box office, reviews, ratings, vote counts, Facebook likes and opening weekends all come after release. They can still appear in descriptive charts, labelled as outcomes.
- As many sources as add something a studio knows in advance (19 Oct). The base list is TMDB's full 2024 dump, joined to The Numbers, IMDb's official datasets, Wikidata, the Movies Dataset, the course file, Lock5 and two FRED series; `data/README.md` lists what each one gives.
- The split follows time (19 Oct): `dev` is 1980-2016, `test` is 2017-2019 (the last three full years before the pandemic), scored once, and `reserve` is 2022-2023, scored once at the end as a post-pandemic check. 2020-2021 are left out because cinemas were closed, and 2024 because its films were still earning when the data was pulled.
- A film counts only if it is a feature film in IMDb, cost at least $1M at 2024 prices and took in at least $10,000 (19 Oct). TMDB holds user-entered junk such as budgets typed in millions.
- Competition is the number of films released within 30 days, measured against that year's median (19 Oct), since the sources cover recent years far more completely.
- A sequel is a film Wikidata numbers second or later in its series, a later film of a Wikidata series already in the table, a film TMDB tags as a sequel, or a title with "2", "III" or "Part Two" (19 Oct). Against TMDB's franchise lists it finds 77% of the sequels, and 71% of what it flags is a sequel there.
- One model, proposed and waiting for a go: a regression on the log of the gross, with an error spread that depends on the budget. Its coefficients say how much each choice moves the box office, and its error spread gives the chance of a flop. The predictions feed a Monte Carlo simulation of film slates.

## Where it stands

Data, split and exploration are done in `movie_studio.ipynb`. What the exploration found on `dev` (7,093 films), ranked by how much it should change the model:

1. A sequel is the strongest pre-release signal. Sequels flop 28-43% of the time against 64-74% for originals in the same budget tier, with a median multiple near 3 in every tier.
2. Budget sets the risk more than the return. The spread of the log multiple falls from 1.95 under $25M to 0.79 above $150M, the only tier where fewer than half flop (47%). The model's error spread has to depend on the budget.
3. Timing matters twice. June and July releases flop 55-58% of the time, September releases 74%, and for films under $150M the most crowded fifth of release windows adds 8-16 points to the flop share.
4. Genre's effect depends on the budget. Animation and adventure pay off only at high budgets, horror only cheap; romance is middling in every tier.
5. A strong record helps a little. A cast whose biggest draw averaged $300M or more flops 59% of the time against 73%, and a director whose earlier films rate 7 or more on IMDb 61% against 66-68%. Counting earlier films shows nothing.
6. Independents flop 74% of the time and spread widest, partly because their budgets are small.
7. Consumer sentiment shows nothing.

### Next

1. Martin's go on the model.
2. Fit it, check it on `test`, then the slate simulation.
3. Book the presentation slot.
4. Write the report and export the notebook as HTML.

## Hypotheses

What the exploration says, as of 19 Oct:

1. A bigger budget lowers the risk more than it raises the return. Holds (finding 2).
2. Genre moves return and risk, and romance is cheap, easy money. Genre matters, but romance is middling, and horror beats it only when the budget is small (finding 4).
3. A known story, a sequel or an adaptation, is safer than an original. Holds strongly for sequels, barely for adaptations (finding 1).
4. The release date matters, through the month and the crowding around it. Holds (finding 3).
5. Experienced, proven talent raises the return. Holds for the quality of the record, not its length (finding 5).
6. Studios differ in what they earn and risk. Independents do worst, partly through budget (finding 6).
7. Films do better when consumers feel good. Does not hold (finding 7).
8. Big and small studios should pick different slates, because they can carry different amounts of risk. Comes from the slate simulation.

## Charts, ranked

Visuals are half the report grade. One accent colour for the group being discussed, grey for the rest, red only for the break-even line, darker blues for bigger budgets. Every chart is a different type and answers one question.

| Rank | Chart | Type | Question |
|---|---|---|---|
| 1 | Budget vs worldwide gross, log axes, with the 1x and 2.5x lines | Scatter | Does spending more pay? Opens the report |
| 2 | Multiple range for originals, adaptations and sequels in each budget tier | Range plot | How much safer is a sequel? |
| 3 | Gross multiple by budget tier | Violin | Does a bigger budget narrow the spread? |
| 4 | Where each column comes from | Stacked bar | How the nine sources fit together |
| 5 | Median multiple vs flop rate by genre, bubble size = number of films | Bubble scatter | Where do romance and horror sit? |
| 6 | Flop rate by genre and budget tier | Heatmap | Which combinations are safest? |
| 7 | Flop rate by release month | Bar | Which months are safest? |
| 8 | Median multiple by crowding, one line per budget tier | Line | Does a crowded window hurt? |
| 9 | Flop rate by talent record | Bar panels | Is proven talent safer? |
| 10 | Studio budget vs spread of results | Bubble scatter | Do studios run different strategies? |
| 11 | Regression coefficients with 95% intervals | Dot-and-whisker | Which choices matter most with the others held fixed? |
| 12 | Predicted range for three example pitches | Interval plot | What should a studio expect from pitch A, B and C? |
| 13 | Slate risk-return frontier | Scatter with frontier line | Which slate fits a big studio, and which a small one? |

Charts 1, 2, 3, 11 and 13 are the core. If pages run out, 4, 7 and 10 go first.

## The slate simulation

A studio has, say, $300M to spend in a year. It can fund one $150M film plus three $50M films, or ten $30M horror and comedy films, and so on. For each slate, draw thousands of outcomes from the model's predicted ranges, turn gross into studio profit with the 2.5 rule, and record the expected profit and the chance the whole slate loses money. Plotted together, the slates form a risk-return frontier like a stock portfolio's, and a big studio that can absorb a bad year picks a different point on it than a small one.

## Limitations to state in the report

1. Box office is not profit. Streaming, TV and home video revenue are missing, and so are real marketing budgets. The 2.5 rule stands in for both.
2. 47% of the budgets and grosses come from TMDB, where users enter them; The Numbers' figures come from studios and trade sources.
3. Actor and director pay is not public, so their earlier box office and ratings stand in for star power. The ratings are IMDb's as of October 2026, so for old films they hold some hindsight.
4. 40% of films have no known MPAA rating, and the sequel flag misses about a quarter of sequels.
5. The sources favour films big enough to be tracked, so small films that vanished are missing and the low-budget tier looks better than it is.
6. These are associations, not causes. A sequel may do better partly because studios only make sequels of films that did well.

## Files

| Path | What |
|---|---|
| `movie_studio.ipynb`, `movie_studio.py` | The analysis, the same notebook in two formats, paired with jupytext |
| `lib/` | Loading, fetching, feature, schema and model code the notebook imports |
| `data/README.md` | Every data file, its source link, units and traps |
| `data/raw/` | The smaller sources as downloaded, plus the saved Wikidata answers |
| `data/interim/` | Extracts of the three large sources, which the notebook reads instead of the raw files |
| `docs/outline-2026-10-19.pdf` | The first outline, as sent to the group |
| `requirements.txt` | The Python packages, at the versions the notebook was run with |

## Running it

You need Python 3.12. Install the packages, then open `movie_studio.ipynb` in Jupyter or VS Code with this folder as the working directory, since the notebook reads `data/` and imports `lib/` by relative path.

```bash
pip install -r requirements.txt
```

The notebook runs from the extracts in `data/interim/`. To rebuild them from scratch, delete that folder and download the three large sources listed in `data/README.md` (about 1.5 GB); the first run then rebuilds the extracts, which takes a few minutes.
