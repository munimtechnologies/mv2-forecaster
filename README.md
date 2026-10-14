# mv2-forecaster

An open-source forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournament (Fall 2026, tournament id 33121, plus the two-week MiniBench rounds).

It was built by **MV2 of Munim, Inc.** Its forecasts are produced entirely by AI, with no human in the loop: nobody looks at or edits a forecast before it is posted. The method is fixed in each release, so the results can be judged against it afterwards.

The Fall 2026 tournament is already running (it opened 2026-09-28) and a skipped question scores 0, so the bot forecasts frequently (every 20 minutes; question windows are a few hours).

## Method (v0.1.1, fixed on 2026-10-13)

Changes since v0.1.0: the model ensemble now prefers each provider's **frontier** model (Anthropic Opus, OpenAI's flagship GPT, Gemini Pro), because across FutureEval's first year the single biggest driver of a bot's score was whether it used a frontier model. The wording that describes the work was also corrected.

For each open question in the tournament that the bot account hasn't forecast yet:

1. **Research.** The first model makes one call with the provider's native web search (OpenRouter `web` plugin, `engine: native`). It is asked for dated facts, the status quo, scheduled events and base rates, and is told not to give a forecast.
2. **Market context** (binary questions only). The bot pulls up to 5 open binary [Manifold](https://manifold.markets) markets that match the title, using Manifold's public search API. The prompt says they may not match the question exactly.
3. **Forecasts.** Every model in the ensemble answers `RUNS_PER_MODEL` times (default 2) at temperature 0.7. By default the ensemble is the newest frontier Anthropic, OpenAI and Google models the endpoint lists (Opus, flagship GPT, Gemini Pro), falling back to the next tier only if a provider's top tier is not listed. Each answer must end in a fixed format:
   - binary: `Probability: NN%`
   - multiple choice: one `Option: NN%` line per option
   - numeric: percentiles 5, 10, 20, ..., 90, 95
4. **Aggregation.**
   - Binary: mean log-odds (the geometric mean of odds), clamped to 2%–98%.
   - Multiple choice: the mean of the probabilities, each floored at 1%, then renormalised.
   - Numeric and discrete: each run's percentiles become a full CDF, and the runs' CDFs are averaged (a mixture, which is wider and safer than a median).
5. **CDF construction.** Percentiles are interpolated linearly, in log space when the question has a `zero_point`. Half of each tail's mass falls inside an open bound, and a closed bound gets 0 or 1.
   - The result is mixed with 2% uniform, which keeps every step at or above 1e-4 (the minimum is 5e-5).
   - An open bound gets at least 0.1% of the mass outside it.
   - No step may exceed the per-bin cap.
   - Every CDF is checked against these constraints before posting (`cdf_problems`).
6. **Posting.** The bot posts one forecast per question, as the bot-tournament rules require, and then a private comment with the aggregate, each run's value, the research summary and one model's reasoning. Metaculus makes these comments public later.

## Rules followed

- No human in the loop. Nobody reruns the bot because of its output or adjusts it after seeing forecasts on open questions. Changes are tested on closed questions or the bot-testing area only.
- One prize-eligible bot. This code is fully open source (MIT) from the start.
- The only outside forecasts it uses are public ones (Manifold).
- AI-made work is disclosed: each forecast's comment states plainly that the forecast was made by AI with no human in the loop, and links here.

## Other venues

The same code can forecast any Metaculus contest that welcomes bot accounts — just add its slug to `TOURNAMENTS`; no code change is needed. Two recurring ones worth entering with this account:

- **ACX / Metaculus Prediction Contest** — the annual [Astral Codex Ten](https://www.astralcodexten.com) × Metaculus contest (~$10,000 pool, ~36 reader-submitted questions, bot accounts explicitly welcome). Each edition is scored from a single **snapshot of standing forecasts in mid-January**, so it should be entered close to that date rather than months ahead. The 2026 edition (slug `ACX2026`) snapshotted on 2026-01-17; the next edition is expected around January 2027.
- **Bridgewater × Metaculus** — an annual accuracy competition (the recent edition ran ~January–March with a $30,000 pool). Check its published contest rules for bot eligibility before entering.

Metaculus uses a **spot peer score** — only each account's last forecast before a question closes counts — so for a snapshot-scored contest the bot should be pointed at it near the deadline, so its standing forecast is fresh rather than months old.

## Run it

```
python mv2_forecaster.py --tournament bot-testing-area --max-questions 3 --dry-run
```

Standard library only (Python 3.10+). Environment variables:

| Variable | Meaning |
|---|---|
| `METACULUS_TOKEN` | Bot account token (Metaculus → FutureEval → participate) |
| `LLM_API_KEY` or `OPENROUTER_API_KEY` | Key for an OpenAI-compatible endpoint |
| `LLM_BASE_URL` | Default `https://openrouter.ai/api/v1` |
| `FORECAST_MODELS` | Comma-separated model ids. If unset, the newest frontier model per provider is used (Opus, flagship GPT, Gemini Pro). Set this to cap cost if the free LLM credits run low |
| `RESEARCH_MODEL` | Defaults to the first forecast model |
| `RUNS_PER_MODEL` | Default 2 |
| `TOURNAMENTS` | Comma-separated tournament ids or slugs. Default `33121,minibench`. Can include other bot-welcoming contests (see [Other venues](#other-venues)) |

Without a token and a key, the bot prints a note and exits 0.

Every forecast is appended to `runs/forecasts.jsonl`. The GitHub workflow does not publish this file: forecast values on open questions stay private, as Metaculus intends. The workflow log shows only question titles. The season is analysed afterwards from the Metaculus API.

GitHub Actions: `.github/workflows/forecast.yml` runs every 20 minutes. Questions open for about 1.5 hours, so this polling interval is enough. The workflow needs the repository secrets `METACULUS_TOKEN` and `OPENROUTER_API_KEY`, and optionally the variable `FORECAST_MODELS`.

## Tests

```
python -m unittest -v
```

The tests cover parsing, aggregation, CDF validity for every bound combination (log-scaled, discrete, very narrow, outside the range) and the forecast flow with a fake model. None of them use the network.

## Results

Results will be posted here after the Fall 2026 season resolves, including the cases where the bot did badly.

MIT licence. By MV2 of Munim, Inc.; the forecasts are made by AI. Contact: munimversion2@gmail.com
