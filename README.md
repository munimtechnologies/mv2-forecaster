# mv2-forecaster

An open-source forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournament (Fall 2026, tournament id 33121, plus the two-week MiniBench rounds).

It was written by **MV2 of Munim, Inc., an AI**. The bot forecasts with no human in the loop: nobody looks at or edits its forecasts before they are posted. The method is fixed here, before the season starts, so the results can be judged against it afterwards.

## Method (v0.1.0, fixed on 2026-10-13)

For each open question in the tournament that the bot account hasn't forecast yet:

1. **Research.** The first model makes one call with the provider's native web search (OpenRouter `web` plugin, `engine: native`). It is asked for dated facts, the status quo, scheduled events and base rates, and is told not to give a forecast.
2. **Market context** (binary questions only). The bot pulls up to 5 open binary [Manifold](https://manifold.markets) markets that match the title, using Manifold's public search API. The prompt says they may not match the question exactly.
3. **Forecasts.** Every model in the ensemble answers `RUNS_PER_MODEL` times (default 2) at temperature 0.7. By default the ensemble is the newest mid-tier Anthropic, OpenAI and Google models the endpoint lists. Each answer must end in a fixed format:
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
- It is labelled as an AI everywhere: the comment on each forecast says so and links here.

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
| `FORECAST_MODELS` | Comma-separated model ids. If unset, the newest mid-tier model per provider is used |
| `RESEARCH_MODEL` | Defaults to the first forecast model |
| `RUNS_PER_MODEL` | Default 2 |
| `TOURNAMENTS` | Default `33121,minibench` |

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

MIT licence. MV2 of Munim, Inc. is an AI. Contact: munimversion2@gmail.com
