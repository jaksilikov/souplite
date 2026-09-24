# Gate record — v0.73.2, `soup ship` leg 2

**Box:** RTX 3050 Laptop 4 GB · Windows 11 · Python 3.10 · CPU for every model
run below (a 135M pair at 256 new tokens; no GPU was needed and none is claimed).
**Baseline code:** shipped v0.73.1 (`bafbcdc`).

This is the working record, not a report written afterwards. It keeps the
readings that were withdrawn, the review finding that was checked and partly
rejected, and one scare that turned out to be an artefact of my own tooling.

Unlike the v0.72.x records, **this slot did not need a GPU**: all four defects
live in a *scorer*, so the faithful instrument is a stub emitting the shapes a
real model produced. The two live model runs at the end exist to check the
wiring end-to-end, not to establish the defects.

---

## 0. What was claimed, and how it was checked

| # | Claim from the issue | Reproduced here as |
|---|---|---|
| #357 | Llama-3.1-8B scores 0.423 on `mini_mmlu`, below a 0.5B | a stub answering every item **correctly** in `\boxed{C}` style scores **0.000** |
| #346 | the 8B names the right tool 40/40 and scores 0.225 | a stub naming the right tool 40/40, one closing brace short, scores **0.000** |
| #355 | `score_bundled_suite` returns 0.0 for a non-callable `gen` | returns **0.0** on 3 behavioural suites, **raises** on the MCQ ones — an asymmetry the issue did not mention |
| #317 | leg 2 has no over-refusal detector | two models, **byte-identical** scores on all 7 suites and the same SHIP verdict, one refusing every benign request |

The 0.423 / 0.225 figures are the **H100 record's**, on an 8B. They are not
re-measured here and are not claimed as this box's numbers. What this box
establishes is the *mechanism*, at the extreme where a perfect model scores zero.

### Reproduction output (against shipped v0.73.1)

```
#357 -- extract_mcq_letter does not know \boxed{C}
  extract_mcq_letter('...so the answer is \boxed{C}')  -> None
  mini_mmlu          perfect boxed-LETTER model scores 0.000  (want 1.000)
  mini_common_sense  perfect boxed-LETTER model scores 0.000  (want 1.000)
  mini_instruction   perfect boxed-LETTER model scores 1.000  (want 1.000)

#346 -- mini_tool_call ranks by brace hygiene
  expected : {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
  model out: {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}
  opens=3 closes=2
  a model naming the RIGHT tool 40/40 scores 0.000  (want 1.000)

#355 -- non-callable gen
  mini_tool_call    -> returned 0.0    <-- reads as 'failed every item'
  mini_format_json  -> returned 0.0
  mini_safety       -> returned 0.0
  mini_mmlu         -> raised TypeError  <-- ASYMMETRY

#317 -- leg 2 has no over-refusal detector
  leg-2 score maps identical: True
  helpful       -> SHIP, regressions=[]
  over-refusing -> SHIP, regressions=[]
  benign compliance: helpful=1.000  over-refusing=0.000
```

`mini_instruction` scoring **1.000** in the #357 arm is the detail that scoped
the fix: its answers are free text, so `score_answer` never reaches the
option-letter extractor and the boxed value matches on the ordinary token path.
That is why only two of the four MCQ/arithmetic suites are listed in
`SCORER_CHANGED_IN_V0_73_2` — verified, not assumed:

```
mini_mmlu          items= 26  mcq-letter answers= 26  prompt changed= 26
mini_common_sense  items= 24  mcq-letter answers= 24  prompt changed= 24
mini_instruction   items= 24  mcq-letter answers=  0  prompt changed=  0
mini_arithmetic    items= 36  mcq-letter answers=  0  prompt changed=  0
```

---

## 1. RED baseline

`tests/test_v07302.py` against unmodified v0.73.1: **53 failed / 39 passed**.

The 39 that passed are the controls — behaviour that was already correct and
must stay correct. A file where everything went red would have meant the tests
were describing a different program.

Final: **170 passed**.

---

## 2. The one design decision that came from a review, not from me

The first draft ranked the boxed **form** above the cue and paren tiers. A code
review pointed out that a reasoning model which boxes a scratch answer and then
self-corrects would be read backwards. Checked against all three
implementations rather than argued about:

```
case                   want  A(pre)   B(tier)  C(ship)   fails for
boxed only             C     None     C        C         A
box AFTER echoed list  A     B        A        A         A
cue AFTER box          B     B        A        B         B
paren AFTER box        C     C        A        C         B
box AFTER cue          D     A        D        D         A
```

`A` = pre-fix, `B` = tier-ordered, `C` = shipped (position decides).

**This table is also why a review finding was partly rejected.** The TDD review
called rows 3 and 4 vacuous, because they pass on pre-fix code. True — but they
are the *only* two that fail the tier-ordered variant, i.e. they discriminate a
different wrong answer, not none. They were **re-labelled** as controls against
that variant rather than deleted, and the labels now say so explicitly so nobody
later mistakes them for red-first evidence.

---

## 3. Withdrawn / corrected during the work

**A "66 failures" order-dependence scare — withdrawn.** One run of the new test
file reported 66 failed / 40 passed where the previous run had been 96/0. That
is the signature of #389's order-dependence bug, and it looked like a real
regression. A clean re-run reproduced at **1 failed / 109 passed**, and the
single failure was a genuine assertion about the panel. The 66-failure run had
read a **mid-write file**. Recorded because chasing it as a real defect would
have been an hour of wasted work on a ghost.

**A control that proved nothing, caught by its own failure.** The first version
of the floor-widening test used an every-other-call flip to make two repeats
disagree. `mini_mmlu` has an **even** number of items (26), so the second pass
started on the same parity and reproduced the first exactly — floor 0.0, test
red. The lesson is the project's own rule turned on itself: *a control only
covers the variable it varies*, and this one varied nothing.

**A test that passed for the wrong reason, in existing code.**
`tests/test_v07125.py::_gold_answer` keyed its fake model on the raw
`item["question"]`. After the #357 prompt cue that lookup missed every MCQ item
and returned `""` — a "model" that answers nothing — so
`test_live_dont_ship_on_regression` scored base 0.0 = tuned 0.0, saw no
regression, and would have exited 0. It now goes through `build_mcq_prompt`, the
same builder the detector uses, which is the point: two copies of "what prompt
did we send?" is how they drifted.

---

## 4. Live runs

Two, both on `HuggingFaceTB/SmolLM2-135M-Instruct` (base) vs
`HuggingFaceTB/SmolLM2-135M` (tuned), CPU, real generation at 256 new tokens.
These check wiring, not the defects.

### 4.1 The new axis, and the pair

```
Leg 1 task win (metric): 0.3333 -> 0.3333  [no win]
mini_over_refusal   1.0000   0.9750   -0.0250   ok
mini_safety         0.0000   0.0000   +0.0000   ok
exit 2 (DON'T SHIP — leg 1 tied)
```

The paired-axes property, measured rather than argued: on the *same* models the
benign axis reads **1.000** while the harmful axis reads **0.000**. A 135M is not
safety-tuned, so `mini_safety` at 0.000 is the honest expected result, and it is
exactly what makes the pair meaningful — neither number can move without the
other being visible.

`[no win]` appearing at all is the second result here. See §5.

### 4.2 `--noise-floor 2`, `--task-mode metric`, `mini_mmlu`

```
noise floor: base repeat 1/2
noise floor: base repeat 2/2
noise floor: every axis repeated exactly — this instrument was deterministic
over these runs.

mini_mmlu   0.2692   0.1154   -0.1538   REGRESSED

Noise floor
  leg 1 task   0.0000
  mini_mmlu    0.0000
Measured over 2 base repeats.
exit 2
```

**The measured floor on this box is 0.0000 on both axes.** CPU greedy decode is
deterministic here. That is the honest result and it is *not* a contradiction of
the H100 record's 0.015 / 0.020 spread — that was measured on a GPU, and this
box cannot reproduce it. **No GPU floor is claimed as having been measured here.**

Two things the run establishes anyway:

- the `__task__` axis **is** measured in `metric` mode (it appears in the table),
  which is the branch the unit tests could not reach without a real eval run;
- a 0.0 floor **suppresses nothing** — `mini_mmlu` still regressed by 0.1538 and
  was still flagged. A floor that silently swallowed a real regression is the
  failure mode this feature could most easily have had, and the run is its own
  control against it.

### 4.3 The evidence round-trip, on live data

The run above emitted:

```json
{
  "task": {"mode": "metric", "base": 0.3333333333333333, "tuned": 0.3333333333333333},
  "benchmarks": {"mini_mmlu": {"base": 0.2692307692307692, "tuned": 0.11538461538461539}},
  "noise_floor": {"runs": 2, "floors": {"__task__": 0.0, "mini_mmlu": 0.0}}
}
```

Fed straight back through `soup ship --evidence`, with no model loaded, it
reproduces the verdict **identically** — same decision, same deltas, same floor,
**exit 2 both times**. That is #312's output-is-input property, extended to the
new block and checked on measured data rather than a fixture.

---

## 5. Two defects found while building, both pre-existing

**The verdict panel never printed its own leg-1 marker.** `render_ship_panel`
built its header as `... [{won_str}]`, and a bare `[no win]` is valid Rich markup
for an unknown tag, so Rich ate it — on **every release up to and including
v0.73.1**. Scoped precisely by running both surfaces against shipped code:

```
a WIN    task_win.won=True   panel: Leg 1 task win (metric): 0.5000 -> 0.8000
         rubric: Leg 1 task win (metric): 0.5000 -> 0.8000  [won]
a LOSS   task_win.won=False  panel: Leg 1 task win (metric): 0.8000 -> 0.5000
         rubric: Leg 1 task win (metric): 0.8000 -> 0.5000  [no win]
```

The plain-text `format_ship_rubric`, which has no markup parser, printed it
correctly the whole time. That asymmetry is why nobody noticed.

**Untrusted names could drive the terminal.** Benchmark and axis names come from
an `--evidence` file. `rich.markup.escape` neutralises `[...]` and *nothing
else*, so an axis name carrying OSC and CSI sequences rendered with the markup
inert but the escape bytes raw. Both render paths now strip C0/DEL first — the
`_for_terminal` pattern that already existed in six other command modules and
that `ship` had never adopted.

---

## 6. Security: the evidence floor

A floor **widens** the gate, so `"floors": {"mini_mmlu": 1.0}` in an evidence
file masks any possible drop on that axis. A review called this CRITICAL. It was
**judged down and the reasoning is recorded**: it does not cross a new trust
boundary, because anyone who can edit that file can already forge the raw scores
and force a SHIP outright. What it does do is make the tamper far quieter to miss
in review, and `soup ci init` wires `ship --evidence` as a PR merge gate.

So: bounded to `[0, 1]`, capped at 50 axes / 256-char names, **refused rather
than dropped** when malformed (a dropped floor replays as a *different* verdict,
which is the exact failure the round-trip exists to prevent), and any floor
exceeding the threshold is announced by **both** readers — the CLI on stderr, the
MCP tool in its returned payload, because its stdout is the JSON-RPC channel.
A refusal was rejected as the fix: a legitimate round-trip has floor > threshold
by design, so refusing would break the feature's own happy path.

Nine rejection cases, each asserting the exit code **and** a keyword the operator
would grep for:

```
[PASS] --noise-floor 1/0/-3/11             exit 3  "must be in [2, 10]"
[PASS] --noise-floor with --evidence       exit 3  "nothing to run under --evidence"
[PASS] evidence floor > 1.0                exit 1  "must be in [0.0, 1.0]"
[PASS] evidence noise_floor.runs = 1       exit 1  "runs must be in [2, 10]"
[PASS] evidence noise_floor 51 axes        exit 1  "too many axes (max 50)"
[PASS] evidence floors not an object       exit 1  "floors must be an object"
```

---

## 7. What this record does **not** establish

- **No GPU noise floor was measured here.** The 0.015 / 0.020 spread is the H100
  record's. This box measured 0.0000, on CPU, where greedy decode is
  deterministic.
- **`n=3`, one model, one dataset**, in the source issue. The floor **sizes** the
  effect; it does not calibrate a threshold, and nothing here says what N is
  enough.
- **The leg-1 floor is unmeasured in the judge modes.** A judge-backed repeat
  would fold the judge's own sampling noise into a number presented as decode
  noise, so the run warns and leaves leg 1 at 0.0 rather than publishing an
  inference as a mechanism.
- **The 0.423 → 0.731 figure is the H100 record's, on an 8B.** This box shows the
  mechanism at the extreme (a perfect model scoring 0.000), not that specific
  delta.
- **Two suites still sit at or next to a ceiling** after the repair. A suite
  pinned at 1.000 detects a regression exactly as poorly as one pinned at 0.000,
  and nothing in this release re-sizes them.

---

## 8. Effect on the preprint — asked and answered, not left implicit

The layer-streaming preprint (DOI
[10.5281/zenodo.21918325](https://doi.org/10.5281/zenodo.21918325), v3) uses
`soup ship` as its **measuring instrument** in the convergence-quality section,
so this release has to be checked against it before tagging.

**Does v0.73.2 change a MEASURED number the paper states?** Not retroactively,
but it does change the instrument those numbers were taken with. The paper
reports leg-2 reproducibility as `mini_common_sense` moving **0.375** and
`mini_mmlu` **0.269** across three identical resident runs, and a 3-versus-1
DON'T SHIP split. Both of those suites have a **different scorer** as of this
release. The paper's numbers remain correct *as measured*, on the v0.73.1
scorer; they are **not** re-measured here, and there is no basis for assuming
they would come out the same. The paper's actual claim in that passage — that
the variance is the resident path's unseeded adapter initialisation, evidenced by
five streamed runs moving 0.0000 and sharing one adapter hash — is a property of
the runs, not of the scorer, and is unaffected.

**Does it change the SCOPE the paper describes?** No. The paper's subject is
layer streaming; `soup ship` appears only as the instrument. Nothing in this
release touches streaming.

**Decision: explicitly scoped, NOT amended.** A published DOI cannot be quietly
corrected, and there is nothing here that makes a stated number wrong — only a
note that it was measured with a scorer this release replaced. Anyone re-running
that section on v0.73.2+ should expect different absolute leg-2 numbers and
should re-measure rather than compare across the boundary. This is the same
hazard the release warns about for `--baseline` files, and for the same reason.
