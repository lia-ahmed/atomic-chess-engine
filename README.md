# Atomic Chess Neural MCTS Engine

An Atomic Chess engine built with PyTorch, a residual policy/value network, PUCT Monte Carlo Tree Search, supervised warm-starting and iterative MCTS self-play. 

The project trains from historical Atomic games, improves through self-play, measures checkpoint-to-checkpoint strength in a controlled internal arena, expose the engine over UCI, and run online through the Lichess BOT API using the maintained 'lichess-bot' bridge.

The engine has been tested both locally and in live Atomic play on Lichess. 

This project was heavily inspired by Google's AlphaZero, and its processes. 

This repository is a cleaned-up public version of an ongoing research/engineering project. Development experiments and intermediate iterations are not all represented in the public history.

---
## Intro

Atomic Chess Engine is a neural Monte Carlo Tree Search engine for Atomic Chess, combining supervised learning from historical games with AlphaZero-style self-play reinforcement learning. It uses a PyTorch policy/value network, PUCT search, and an iterative self-play training pipeline, and can be run as a UCI engine or deployed as a Lichess bot.


## Features

- Atomic Chess move generation through 'chess.variant.AtomicBoard'
- 14-plane board representation 
- residual convolutional policy/value network
- historical next-move policy pretraining
- historical game-result value warm-starting
- PUCT Monte Carlo Tree Search
- AlphaZero-style self-play targets from MCTS visit counts
- iterative policy/value reinforcement training
- CUDA mixed-precision training
- sharded self-play generation
- internal head-to-head arena with paired openings
- bootstrap confidence intervals for relative Elo estimates
- UCI engine interface
- clock-aware search budgeting for online play
- Lichess BOT integration through `lichess-bot`
- deployment/rating provenance logs

---

## Architecture

```text
Historical Atomic games
        |
        v
validated per-ply Parquet
        |
        v
14 x 8 x 8 board representation
        |
        v
historical policy/value warm start
        |
        v
PolicyValueNet
        |
        +------------------------------+
        |                              |
        v                              v
PUCT MCTS                         value prediction
        |
        v
self-play games
        |
        v
MCTS visit-policy targets + final-result value targets
        |
        v
self-play trainer
        |
        v
new checkpoint
        |
        +--> internal arena --> promote/reject
        |
        +--> UCI --> lichess-bot --> Lichess Atomic play
```

---

## Board representation

Each position is encoded as a 'uint8[14,8,8]' tensor:

```text
0-5    White P, N, B, R, Q, K
6-11   Black P, N, B, R, Q, K
12     side-to-move plane
13     Atomic explosion-threat mask from legal captures
```

The tensor orientation is rank/file based with 'a1' at `[*,0,0]`.

The current policy action space is a dense index over normalized UCI move strings
observed in the historical dataset. This keeps the policy head compact but is a
known limitation: it is not a complete theoretical move encoding. Self-play
tracks unknown selected moves and dropped visit-policy mass so this limitation
can be monitored.

---

## Model

'PolicyValueNet' is a small residual CNN with a shared trunk and two heads:

```text
14x8x8 input
    |
convolution stem
    |
residual blocks
    |-------------------|
    v                   v
policy head          value head
    |                   |
action logits       scalar [-1,1]
```

The value is always interpreted from the perspective of the side to move:

```text
+1   eventual win
 0   draw
-1   eventual loss
```

---

## Training strategy

The training pipeline uses three distinct sources of supervision.

This pipeline is heavily derived from AlphaZero's pipeline. 

### 1. Historical move policy

Historical positions are paired with the next move from the same game. The
original per-ply data convention stores the position after a move, so a policy
example pairs the state at ply `N` with the move recorded at ply `N+1`.

### 2. Historical value warm start

The same historical positions also receive a value target derived from the final
game result, converted into the perspective of the player to move.

This gives the value head useful signal before expensive MCTS self-play begins.

### 3. MCTS self-play

For self-play positions:

- policy target = normalized MCTS root visit distribution
- value target = final game result from that position's side-to-move perspective

Self-play uses exploration. Evaluation and rated play do not.

---

## Repository layout

A typical checkout is organized as:

```text
.
|-- src/
|   |-- representations.py
|   |-- dataset.py
|   |-- models.py
|   |-- historical_pv.py
|   |-- train_historical_policy_value.py
|   |-- mcts.py
|   |-- self_play.py
|   |-- trainer.py
|   |-- arena.py
|   |-- ratings.py
|   |-- time_control.py
|   |-- uci_engine.py
|   |-- lichess_rating.py
|   `-- log_deployment.py
|
|-- scripts/
|   |-- verify_selfplay.py
|   `-- uci_smoke_test.py
|
|-- tests/
|
|-- integrations/
|   `-- lichess/
|       |-- config.atomic.snippet.yml
|       `-- setup_lichess_bot.ps1
|
|-- action_map.json
`-- requirements.txt
```

---

## Installation

Create a Python environment and install the project requirements:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your platform if GPU use is desired, then verify it:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Data preparation

The project expects a per-ply Parquet dataset containing Atomic positions,
metadata, final result, FEN and flattened board planes.

Build the action map from an existing per-ply dataset:

```powershell
python src/representations.py `
  --build-action-map data/per_ply.parquet `
  --out action_map.json
```

Validate stored planes against FEN-derived representations:

```powershell
python src/representations.py `
  --validate-parquet data/per_ply.parquet `
  --sample-size 1000
```

If rebuilding the historical dataset from DuckDB, see `extract_per_ply.py` and
its `--help` output.

---

## Historical policy/value warm start

The preferred historical warm-start stage uses the policy/value architecture and
trains both heads jointly:

```powershell
python src/train_historical_policy_value.py `
  --data-dir data `
  --action-map-path action_map.json `
  --init-checkpoint ckpts/phase4/init.pt `
  --ckpt-dir ckpts/phase4/historical_pv `
  --epochs 3 `
  --batch-size 256 `
  --lr 0.0003 `
  --num-workers 4 `
  --device cuda
```

The original policy-only pretraining and policy-to-policy/value conversion are
kept as bootstrap/reproducibility stages where required.

---

## Self-play

Generate MCTS self-play:

```powershell
python src/self_play.py `
  --net ckpts/phase4/<checkpoint>.pt `
  --action-map-path action_map.json `
  --out-dir selfplay/canonical/iter_000 `
  --games 1000 `
  --simulations 200 `
  --c-puct 1.5 `
  --temperature 1.0 `
  --temperature-moves 20 `
  --iteration 0 `
  --device cuda
```

Validate the generated shards:

```powershell
python scripts/verify_selfplay.py selfplay/canonical/iter_000 --expected-games 1000
```

Self-play output is JSONL and is intentionally separate from arena/Lichess
evaluation data.

---

## Reinforcement/self-play training

Train a new candidate from MCTS-generated targets:

```powershell
python src/trainer.py `
  --data-dir selfplay/canonical/iter_000 `
  --action-map-path action_map.json `
  --init-checkpoint ckpts/phase4/<source>.pt `
  --ckpt-dir ckpts/phase4/rl_iter_001 `
  --iteration 1 `
  --epochs 3 `
  --batch-size 256 `
  --lr 0.0003 `
  --num-workers 4 `
  --device cuda `
  --rebuild-index
```

The trainer uses:

```text
policy loss = cross entropy against the MCTS visit distribution
value loss  = MSE against the final game outcome
```

Checkpoints include model configuration and an action-map digest for compatibility
checking.

---

## Internal strength evaluation

A candidate should be evaluated against the current baseline under identical
search conditions before promotion.

Example:

```powershell
python src/arena.py `
  --candidate ckpts/phase4/rl_iter_001/best.pt `
  --baseline ckpts/phase4/historical_pv/epoch_003.pt `
  --action-map-path action_map.json `
  --out-dir evaluation/arena/rl001_vs_historical `
  --games 200 `
  --simulations 200 `
  --opening-plies 4 `
  --device cuda
```

Arena games use:

- zero-temperature move selection
- no root Dirichlet noise
- equal MCTS budgets
- paired random openings with colors reversed

Outputs include JSONL results, PGN and a summary with an internal relative Elo
estimate and bootstrap confidence interval.

Recompute rating statistics independently with:

```powershell
python src/ratings.py evaluation/arena/rl001_vs_historical/games.jsonl
```

The internal Elo delta is for checkpoint comparison only; it is not an external
chess rating.

---

## UCI engine

Smoke-test a checkpoint through the UCI boundary:

```powershell
python scripts/uci_smoke_test.py `
  --checkpoint ckpts/phase4/rl_iter_001/best.pt `
  --action-map-path action_map.json `
  --device cuda `
  --movetime-ms 500
```

Run the engine directly:

```powershell
python src/uci_engine.py `
  --checkpoint ckpts/phase4/rl_iter_001/best.pt `
  --action-map-path action_map.json `
  --device cuda
```

The engine advertises `UCI_Variant=atomic` and supports ordinary UCI position and
clock commands required by the Lichess bridge.

---

## Time management

Online play uses a simple clock-aware simulation budget. The allocator considers:

- remaining clock
- increment
- fixed overhead allowance
- minimum clock reserve
- minimum/maximum move time
- measured simulations per second

The search-speed estimate is updated online using an exponential moving average.

Current MCTS is not interruptible mid-search, so time management is deliberately
conservative: it chooses a search size expected to fit inside the target budget
rather than enforcing a hard deadline.

---

## Lichess BOT integration

The repository uses the maintained external `lichess-bot` project instead of
reimplementing Lichess event streams and challenge handling.

The boundary is:

```text
PolicyValueNet + MCTS
        |
        v
src/uci_engine.py
        |
        v
lichess-bot
        |
        v
Lichess BOT API
```

`integrations/lichess/config.atomic.snippet.yml` contains the Atomic-specific
engine/challenge settings to merge into the current `lichess-bot` default config.

Never commit a Lichess access token. The integration is designed to read it from:

```text
LICHESS_BOT_TOKEN
```

Record a deployment:

```powershell
python src/log_deployment.py `
  --checkpoint ckpts/phase4/rl_iter_001/best.pt `
  --action-map-path action_map.json `
  --label lichess_v001
```

Record a public Atomic rating snapshot:

```powershell
python src/lichess_rating.py <BOT_USERNAME> `
  --out evaluation/lichess/ratings.jsonl
```

Lichess uses Glicko-2. The external Lichess Atomic rating is therefore separate
from the project's internal relative Elo measurement.

---

## Testing

Current test groups cover representation, historical datasets/training, policy
and value networks, checkpoint transfer, MCTS, self-play, rating math and timing.

Run the public test suite with:

```powershell
python -m unittest discover -s tests -v
```

---

## Reproducibility

The project uses:

- deterministic game-level train/validation splits
- explicit random seeds
- action-map SHA256 compatibility checks
- checkpoint provenance
- immutable iteration directories
- per-epoch metric logs
- self-play validation
- deployment checkpoint/action-map hashes

For meaningful playing-strength comparisons, keep search settings fixed across
candidate and baseline engines.

---

## Current limitations / planned improvements

- MCTS leaf inference is currently unbatched.
- MCTS cannot currently be interrupted mid-search for a hard time deadline.
- `self_play.py` does not yet have native crash-resume support.
- The observed-UCI action vocabulary is not a complete move encoding.
- Large self-play campaigns are compute-bound by sequential tree search rather
  than neural-network training.
- Lichess playability is limited by simple timing configuration. 

Likely next engineering improvements include batched inference, resumable self-play
shards, hard search deadlines and eventually a complete fixed action encoding.

---

## Evaluation philosophy

Two complementary measurements are used:

```text
Internal arena
    -> controlled checkpoint-to-checkpoint strength measurement

Lichess Atomic rating
    -> public external milestone against an independent player pool
```

Training metrics are useful diagnostics, but playing strength is ultimately
measured by games.

---

## Data and model artifacts

Raw Lichess game archives, large databases, generated Parquet datasets, self-play
buffers and model checkpoints can be very large and are not expected to live in
the Git source tree. Document or publish those artifacts separately when needed.

---

## Project status

Implemented and working:

- historical Atomic data pipeline
- neural policy/value model
- PUCT search
- MCTS self-play
- iterative policy/value training
- internal arena evaluation
- UCI engine
- Lichess BOT deployment path

The project is under active development and should be treated as an experimental
engine rather than a finished production chess engine. Use at your own risk. 


This repo contains some of the utilised code, but not all (e.g. current weights are not included).

I acknowledge the use of Large Language Models in the coding process, but all code is checked and any errors are my own. And I also acknowledge that Copilot is a stupid dumb lying no good LLM. 