# Red Lite dev52 — activation steering

Status: on branch `dev52/steering`, measured on the Apple M4 Max 48 GiB on 2026-10-03. Roadmap Phase 4, approved by
the project owner. The idea and its use follow antirez's ds4 (`docs/RESEARCH_2026_10.md` section 6).

## What it does

**The mechanism.** A steering vector is `hidden` (2048) float32 values. While steering is on, every decoded token
adds `strength × vector` to the residual stream at the input of the chosen layers, before their RMSNorm. This shifts
the model's internal state along a direction: a style, a language, a topic.

**Where it applies:**
- **Steered:** every decode path.
  - The bounded-cache synchronous path.
  - The GPU-routed path at full residency.
  - The 2-row MTP verify, on both rows.
  - The CPU oracle.
- **Not steered:** the batched prefill.
  - So that the first answer token is steered too, the last prompt token goes through an ordinary steered step
    when steering is on.
- **Steering the start of an answer.** `--steer-tokens N` steers only the first N tokens of each answer, then turns
  steering off. As ds4 observed, an autoregressive model keeps the direction it started in.

**Options and commands:**

```bash
python3 scripts/dev/steer_extract.py MODEL --pos POS.txt --neg NEG.txt --layer 24 --out v.f32
./bin/redlite chat --steer v.f32 --steer-layers 12-23 --steer-strength 0.3 [--steer-tokens 40]
#   in the chat: /steer 0.5 changes the strength, /steer 0 turns it off, /steer prints it
.deps/redmetal/redlite-generate MODEL --steer v.f32 --steer-layers A-B --steer-strength S --steer-tokens N --prompt "..."
```

**Extracting a vector.** `scripts/dev/steer_extract.py` takes two files of user messages, one per line. Each message
goes through the chat template and `redlite-engine logits`. The script reads the residual stream entering `--layer`
at the last prompt position, and the vector is mean(positive) − mean(negative). No Python runs at chat time; the
vector is a plain file.

**API.** `rl_engine_set_steering`, `rl_engine_set_steering_strength`, `rl_engine_load_steering`. The Metal kernel is
`rl_steer_add`, one fused multiply-add per element.

## Validation (M4 Max 48 GiB, IQ2_XXS)

| check | result |
|---|---|
| CPU oracle vs Metal, steering on (random vector, layers 10–30, strength 3), bounded cache | `MULTI-TOKEN ENGINE PARITY: YES`; logits max diff 5.7e-6, 0 router mismatches; the argmax changes vs unsteered (101924 vs 11), so the vector takes effect |
| same, GPU-routed path (`--cache-mib full --repeat 2`) | `MULTI-TOKEN ENGINE PARITY: YES` |
| steered answer with and without MTP (pirate vector, layers 12–23, strength 0.3, first 40 tokens, 120 tokens greedy) | identical byte for byte; acceptance 0.710 |
| steering off | unchanged: no code path runs while the strength is 0; `quick_parity.sh` passes |

**Regression check.** `regress_m4.sh` has a new check, `engine.parity.steered`: the parity above with a fixed random
vector.

**A note on the dumps.** With steering on, the per-layer dumps of the two backends differ by exactly the steering
delta (0.52 = 3 × the vector's largest value). The two backends record layer outputs on different sides of the next layer's
steering (not investigated further). The final norm and logits agree to 1e-6.

## Example: a "pirate" direction

Twelve pirate-speak requests against the same twelve questions asked plainly (`tests/fixtures/steer/`), extracted at
layer 24. |v| = 4.35, against 10.1 for the mean residual norm there. Prompt: "Explain in two sentences how a
bicycle works.", greedy:

| layers, strength | answer starts with |
|---|---|
| none | "A bicycle works by converting the rider's pedaling motion into forward motion through a system of gears…" |
| 20–27, 0.4 | "…locking the gears so the whole rig stays upright. The gears whisper through the air…" |
| 12–23, 0.4 | "When you pedal the crank, the chain whips the gears—click-clack—right foot stompin' the axle, the wheel spins wild, the rim whirrin'…" |
| 16–31, 1.0 | broken text (the vector is added 16 times at full strength) |

With "Describe a storm at sea" (layers 12–23, 0.3, first 40 tokens), the answer turns into a sea tale: "sails torn to
shreds, masts groaning in the gale…".

**Typical doses.** 0.2–0.5 over 8–12 layers. Strength 1 over 16 layers breaks the text: the shift becomes
several times the residual norm.

## What is not included

- **No pre-made vectors**, beyond this example's prompt files. Users build their own from their own prompt sets.
  The project does not ship vectors that switch off the model's refusals.
- The prefilled conversation is in dev52b, below.

## dev52b — prefilled conversation (`--history FILE`)

`redlite chat --history FILE` (and `redlite-generate`) reads a text file of past turns before the first message:

```text
user: Who are you?
assistant: Arr, I be Captain Redbeard, the saltiest pirate on the seven seas! I answer every question in pirate speak.
user: Do you always talk like that?
assistant: Aye, matey, always!
```

- **Format.** A line starting with `user:` or `assistant:` opens a turn; the lines after it continue it. The turns
  are rendered in ChatML between the system prompt and the new message, so the model takes them as its own past.
  The steering vector is not touched.
- **After `/reset`** the history is read again.
- **Bad files are refused** with the reason: text before the first turn, or no turns at all.

**Example** (IQ2_XXS, greedy):
- "Hi, how are you today?" with this history: the prompt is 80 tokens instead of 15, and the answer starts "Ahoy,
  friend! I be Captain Redbeard, freshly returned from a nap on the Jolly Roger…".
- Without the history: "Hi! I'm doing great—thanks for asking!…"
- On a plain task ("Explain in two sentences how a bicycle works.") the same history did not change the answer's
  style: the task wins over the persona.

## Scope boundary

- Measured on the M4 Max only, with IQ2_XXS. The M4 Pro runs the same code paths: the kernel and the hooks have no
  machine-specific part.
- The server (`redlite-server`) does not take steering options yet.
- Quality effects (which doses keep answers useful) are only the observations above, not measured.
