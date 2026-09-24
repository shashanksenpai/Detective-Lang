# BACKLOG — task bucket list

Things we have agreed to do **later**, plus an honest list of rough or unchecked areas. This is not the
roadmap (that is in `CLAUDE.md`); it is the running list of loose ends.

**How to keep it current:** when a limitation, an unverified area or a good idea turns up, add a line here
with the next free ID. When something is fixed, tick it and move it to *Done* with the date and what fixed
it. Say plainly what was and wasn't verified — "loads without errors" is not "looks right".

Last updated: 2026-09-24

## Priority order (set 2026-09-24)

Autonomous work goes first; anything that needs a human decision stops for it.

1. **Hygiene** - S-1 static-mount exposure (done), tests skip without the model (done), docs kept honest (done).
   Still open and waiting on the owner: **H-1 LICENSE**, **S-2 CORS / no auth** (compatibility call).
2. **Parser fidelity - F-01, F-02, F-03.** Everything downstream (the chat reader, contradictions on real chats)
   is only as good as the import, and real exports hit these first. No sign-off needed.
3. **N-1 design** - a plan for approval, not code: the method is deliberately undesigned (CLAUDE.md, Phase 6).
4. **N-1 implementation**, then **N-2 chat reader** (needs step 2).
5. **N-3** waits for the owner's specifics.

---

## Next up (agreed features)

- [ ] **N-1 · Contradiction detection (roadmap Phase 6).** Top priority. Needs a design session before any
  code: what counts as a contradiction, deterministic vs LLM for each kind, how a finding cites its two
  statements, how it says "uncertain", and how a human confirms it (a confirmed pair could be pinned as
  evidence). Evaluate against `sample_leak_case_key.json` — recall on the high-severity items C01–C16,
  precision against the decoys D01–D05. Constraints are in the CLAUDE.md roadmap entry.
- [ ] **N-2 · Chat reader — read any imported chat in full, exactly as it was.** A per-source transcript
  view: every message in order, day separators, sender names, timestamps, jump-to-date, in-chat search,
  pinned messages marked in the margin, reachable from the workspace and from any message anywhere (search
  hit, timeline day, pin, contradiction, board).
  **Depends on parser fidelity — "as it is" is not possible today:** the WhatsApp parser drops continuation
  lines of multi-line messages (F-02) and drops system lines (joined/left/encryption notice/deleted
  message), and `<Media omitted>` is stored as if it were a spoken message (F-03). Decide first whether to
  keep every raw line (including system events, as non-message rows) so the reader can show the original.
  Also keep the original file reachable from the source (uploads are kept; the seeded demos point at the
  bundled samples).
- [ ] **N-3 · Investigator-style presentation (roadmap Phase 7).** Details to come from the user; do not
  design ahead of them. The investigation board was reworked on 2026-09-21 as a first step (see *Done*).

---

## Fine-tunings for features not fully checked

### Import / parsers
- [ ] **F-01 · WhatsApp export formats.** The line regex only accepts the en-US style (`H:MM AM/PM`
  upper-case, 2- or 4-digit year, ` - ` separator). Other locales (24-hour, lower-case am/pm, narrow
  no-break space before AM/PM on newer exports, `[dd/mm/yy, hh:mm:ss]` iOS style, `dd.mm.yy`) fail with
  "No messages parsed". Real exports from an Indian-locale phone are the likely first thing to break.
- [ ] **F-02 · Multi-line WhatsApp messages.** Continuation lines are dropped (only the first line of the
  message is kept). Changing this must keep `seq` stable for existing sources (pins and the timestamp
  backfill match by message id / `seq`).
- [ ] **F-03 · Media, system and deleted lines.** `<Media omitted>` counts as a message in profiles,
  attribution, sentiment, search and the timeline. It should be flagged and excluded from stylometry,
  embeddings and mood. Same for "This message was deleted" and similar.
- [ ] **F-04 · Ambiguous WhatsApp date order** is read month-first with a visible note; there is no way to
  say "these are day-first" on upload.
- [ ] **F-05 · Timezones.** Instagram times are UTC, WhatsApp/Telegram are the device's local time, so a
  cross-platform case is off by the UTC offset. Needs a per-case timezone.
- [ ] **F-06 · Instagram and Telegram parsers** have only ever seen the hand-written samples, never a real
  export.
- [ ] **F-07 · One bad date line undates a whole WhatsApp file** (the date-order inference needs every
  line valid under one order). Consider undating just the bad lines.
- [ ] **F-08 · Server start blocks while seeding** (~40 s the first time with the paper-leak demo). Seed in
  the background and let the status badges show it.
- [ ] **F-09 · Dev ergonomics.** `uvicorn --reload` hangs on this Windows setup; restarts are manual.
- [ ] **F-10 · No way to remove a *ready* source or delete a case.** Deliberately deferred: it has to
  untangle people, merge suggestions, pins and board layouts.

### Workspace (search / timeline / evidence)
- [ ] **F-11 · Sentiment on Hinglish — follow-ups.** The big fix is done (see *Done*); what is left is F-21 to F-30 below.
- [ ] **F-12 · Timeline polish.** A mood cell built from one message looks as certain as one built from
  fifty (fade by count?); no zoom or brush for very long ranges; only the day columns are keyboard-focusable;
  checked at 400 px width for overflow but never on a real phone.
- [ ] **F-13 · Search.** Plain substring/phrase only (no semantic search, no typo tolerance, no saved
  searches); it scans in Python, fine at chat-export size but not at 100k+ messages.
- [ ] **F-14 · Evidence.** Only messages can be pinned — not a contradiction, an attribution result or a
  board relationship. No export / print of the evidence list. Notes have no history.
- [ ] **F-15 · Automated API tests.** Pins, search, context, timeline, board layout and source removal were
  verified by hand against a scratch server; there are no API tests. Needs DB isolation (make
  `DATABASE_URL` overridable) and pytest fixtures.

### Attribution / identity
- [ ] **F-16 · Uncertainty thresholds are calibrated for 3-person cases.** On the 9-person paper-leak case
  top-1 accuracy is 46.9% (VADER fallback) / 44.1% (Hinglish model) - re-measured 2026-09-24, the 49.7% first
  recorded here is stale - and the engine commits on 0.7% / 0.0% of test messages. (Improvement Stage item 1.)
- [ ] **F-17 · spaCy syntax signal** (Improvement Stage item 2).
- [ ] **F-18 · `DetectiveEngine` treats the last message of one source and the first of the next as
  adjacent** when it counts turn-taking (`transition_counts`). The board's exchange counting had the same
  flaw and was fixed on 2026-09-21; the engine still has it.
- [ ] **F-20 · Test speed.** `test_graph_analysis.py` takes ~1 minute because importing `graph_analysis`
  pulls in the whole ML stack. Move the pure rules (`tone_for`, exchange counting) into a light module.
- [ ] **F-21 · Hinglish sentiment: known misses.** On 60 unseen plain-Hinglish probes 10 are wrong, e.g. "bhai tu toh
  chha gaya", "hum jeet gaye!! kya match tha", "main bahut dukhi hoon, koi meri baat nahi samajhta"; plain "yeh acha hai" is
  scored slightly negative (n-gram bias from "lag raha", "gaya"). Two are marked `xfail` in `test_hinglish_sentiment.py` —
  remove the marker when a retrain fixes one. More labeled chat-register data is the likely fix, not more rules.
- [ ] **F-22 · Negative recall on chat.** On the (English-heavy) demo chats the model catches 47% of negatives vs VADER's 63%
  (19 test negatives — noisy). Try a higher negative-class weight or a chat-domain calibration, judged on real chats.
- [ ] **F-30 · Emoji are register-specific, and the trade cost something.** Emoji features are learned only from chat-register
  rows, so Hinglish YouTube comments (where emoji carry signal) dropped from 67.5 to 62.2 macro-F1. If the app ever scores
  comments/social text, give that register its own emoji weights. Also: the curated emoji list is one person's reading of
  chat emoji - check it against the user's own chats (F-25).
- [ ] **F-23 · Devanagari and mixed-script text** (Hindi typed in Devanagari, or both scripts in one message) is not handled by
  the lexicon or the model. Needs transliteration or a multilingual model.
- [ ] **F-24 · Sarcasm, irony, and "tense" that isn't negative words.** A cold accusation ("You wrote 'hostel. room.' in the
  group. Was that accurate?") reads neutral, so the board's tense tone can't see confrontation. Pragmatic cues (quoting the
  other person, "you said", pointed questions) belong with contradiction detection (N-1).
- [ ] **F-25 · No validation on real chats.** Every chat number is from synthetic demo chats with one labeler. Add a small
  in-app flow to label ~50 of the user's own messages and feed them to `eval_sentiment.py` (the harness already takes a labeled
  set), so the model can be judged and retrained on the user's real text.
- [ ] **F-26 · Hinglish word lists (topics, question/command/connector/pronoun words, dossier stop-words) are unproven.** They
  showed no effect on the English-heavy demos (paired attribution check, 910 predictions). Measure on a real Hinglish chat.
- [ ] **F-27 · `explain()` is not shown anywhere.** The scorer can say which n-grams and lexicon words drove a message's score;
  surface it as a "why this reading" hover in search results, the timeline day view and the future chat reader.
- [ ] **F-28 · Tone thresholds and mood bins were set under VADER** (warm > +0.15, tense < -0.05, timeline bins at 0.05/0.2/0.4).
  They were only re-checked for a similar score distribution, not re-fit. Fit them once there is labeled data (Improvement Stage).
- [ ] **F-29 · The sentiment model needs the network to retrain** (downloads ~8 MB of public data, ~80 s) and its training data
  has mixed licences (SentiMix OpenRAIL, YouTube CC-BY-4.0, English tweets unstated). Check before redistributing the model file.
- [ ] **F-19 · Identity soft tier** is disabled until there is more data per person (see
  `eval_identity.py`).

### Investigation board (after the 2026-09-21 rework)
- [ ] **B-1 · What counts as an "exchange".** It is two consecutive messages by different senders, with no
  time limit: a reply the next morning counts the same as one seconds later. Now that `sent_at` exists,
  require a maximum gap.
- [ ] **B-2 · Group vs DM.** All sources in a case are pooled for the edges; a DM exchange and a group
  reply weigh the same. Context-split edges would fit the "don't average group and DM" principle.
- [ ] **B-3 · Layout is per case, not per person-merge**, and is stored by raw sender name.
- [ ] **B-4 · Not checked:** the board on a real phone or touch screen (drag and pinch); with 20+ people
  (the starting ring, label placement and the one-blur-per-group glow are only proven at 3 and 9 people,
  and blur cost with hundreds of beams is unmeasured).
- [ ] **B-6 · Only 1 of 26 called links is "tense" in the paper-leak demo (16 warm, 10 neutral)** - the tone comes from the Hinglish
  model, and a friendly group with cold accusations reads neutral (F-24). The filter is honest but the demo doesn't show it off. Also worth a laser-colour check with the
  user: blue = warm / red = tense matches the timeline, but amber for warm may read more naturally.
- [ ] **B-7 · No board interaction for pins/evidence yet** - a relationship can't be pinned (see F-14).
- [ ] **B-5 · Case-scoped only.** The cross-case combined board mentioned in the original Phase 4 plan is
  still not built.

### Pages not visually reviewed
- [ ] **V-1 · `person.html` (dossier) and `combined_dossier.html`** with 9 people and 5 DMs — only "loads
  without errors" is confirmed for the paper-leak case, not how it reads.
- [ ] **V-2 · `detective_lang.html` (investigate)** with 9 similar speakers — mostly "uncertain" (see F-16);
  the UI for that state hasn't been reviewed.
- [ ] **V-3 · `merge_review.html`** and `cases.html` at phone width.

### Security and repo hygiene (found 2026-09-24 while publishing the repo)
- [ ] **S-2 · No authentication, and CORS is `allow_origins=["*"]`** (`server.py`). Every API route is open to
  anything that can reach the port, and the wildcard means a web page open in the same browser could in principle
  read the local API (browsers restrict this for localhost to varying degrees; **not tested**). Recommended fix:
  allow only the server's own origins (`http://127.0.0.1:8000`, `http://localhost:8000` - the pages already
  hardcode `API_BASE` to :8000) and reject unexpected `Host` headers. **Not changed yet** because it would break
  opening the pages from `file://` or from another port; needs the owner's OK. Do this before any shared use.
- [ ] **H-1 · No LICENSE file** in the repo (public since 2026-09-24), so nobody has permission to reuse the code.
  The owner has to choose the licence; not picked on their behalf.
- [ ] **H-2 · The trained sentiment model is deliberately not committed** (mixed training-data licences, F-29).
  Decide whether to publish it (e.g. as a release asset, with attribution) or keep rebuilding it from source.
- [ ] **H-3 · README figures are copied by hand** from `eval_*.py` / `sentiment_metrics.json`. Re-check them after
  any change to the signals, weights, thresholds or the sentiment model (a stale figure already turned up: F-16).

### Infrastructure (deferred by the roadmap)
- [ ] **I-1 · SQLite → Postgres + pgvector, thread pool → Celery + Redis** — only if real usage volume
  justifies it.

---

## Ideas (unscheduled — chosen for how well they build on the chat reader)

- **Two-window "same moment" view.** Show what a person said in the group and in each DM *at the same time*,
  side by side, scrolling together by timestamp. This is where a suspect's different stories to different
  people become visible at a glance, and it feeds contradiction detection directly.
- **Claims layer.** Highlight statements that assert something checkable ("I was in my room by 8:30", "I've
  never been inside that shop", "it went to Lalit") as chips in the transcript; each chip is a candidate
  input for contradiction detection and can be pinned.
- **Alibi map.** For one person and one night, plot every claim about where they were against every piece of
  evidence (messages sent, photo times, a gate register), on a single time axis.
- **Replay.** Play a chat back at adjustable speed with a scrubber whose height is the activity; long
  silences and sudden bursts stand out.
- **Entity index.** Every mention of a name, amount (₹4k), place ("gate 2") or time, grouped and clickable,
  with a running total for money.
- **Reading modes.** "As exported" (raw lines) vs "conversation" (bubbles), and a diff between the two —
  useful to prove nothing was lost on import.
- **Chain of custody.** Store a SHA-256 of every imported file and show it with the source, plus an export of
  the pinned evidence with those hashes.
- **Margin markers.** In the reader, mark pinned messages, contradiction endpoints and board-selected people
  in the margin, and jump between them.

---

## Done

- 2026-09-21 · **Investigation board rework** — see *Investigation board* in CLAUDE.md (larger canvas,
  drag-anywhere with saved positions, laser edges, glowing selection, tone/exchange filters, readable
  labels, per-source exchange counting, an explicit "unclear" tone for thin evidence).
- 2026-09-22 · **Hinglish sentiment (F-11)** — VADER replaced by `hinglish_sentiment.py` after measuring it on real data (SentiMix
  macro-F1 46.9 -> 68.7, English 65.5 -> 72.5, unseen plain-Hinglish probes 47% -> 83%, blind-labeled chat test
  53.9 -> 64.0 (likely overstated, F-25); Hinglish YouTube comments 40.6 -> 62.2, a known cost, F-30). See "Hinglish sentiment" in CLAUDE.md for what worked, what did not, and the limits.
- 2026-09-24 · **S-1 · Static mount exposed `detective.db`, `uploads/`, the source and `.git/`** - `server.py` served the whole
  project directory. Found on a clean clone (`GET /detective.db` -> 200); fixed by `ui_static.UIStaticFiles`, which serves
  only top-level `.html` pages. Verified on a real running server (pages, `/docs`, API 200; database, source, `.git`,
  `uploads/`, traversal 404) and by `test_ui_static.py`.
- 2026-09-24 · **Fresh clone failed 16 tests** because the model is not committed - the 13 model-behaviour tests now skip
  with the instruction to run `python eval_sentiment.py --retrain` (fresh clone: 154 passed, 29 skipped; with the model:
  181 passed, 2 xfailed). A model file with the wrong feature version still fails rather than skips (verified).
- 2026-09-24 · **Stale figure corrected** (F-16 / CLAUDE.md): paper-leak top-1 is 46.9% / 44.1%, not 49.7%.
