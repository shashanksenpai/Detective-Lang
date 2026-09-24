# Detective Lang

## What this is
A speaker-attribution tool for chat exports (WhatsApp, Instagram, Telegram, etc.). Given a sentence, it returns a ranked list of who in a chat most likely said it, with confidence scores — plus cross-chat identity resolution so the same person is recognized across different platforms/apps and different behavioral contexts (group vs 1:1).

## Current state
The attribution engine (`detective.py`, `person_profile.py`) is unchanged in substance — six-signal
scoring (topic embeddings, stylometry, emotion, topic-domain category, syntax, discourse), softmax
confidence with dynamic weight redistribution, uncertainty thresholding (<35% top confidence or
<8pt margin), and the deliberate refusal to use demographic signals — but as of **Phase 1** it now
runs on a real **case-backed data model** instead of two hardcoded files:

- Parses WhatsApp-style exports (64 synthetic messages across 3 senders in the seeded demo case,
  spanning studies/fun/food/work/logistics topics) — via a **parser registry** (`parsers/`), not a
  single hardcoded function
- Builds per-sender profiles across the six signals described above (see `detective.py` docstrings
  for the reasoning behind each)
- `category_features`/`dominant_category` return "no signal" (`None`) rather than fabricating a
  label when zero lexicon terms match, and gender/demographic signals are explicitly out of scope
  — see the reasoning in `detective.py` and `person_profile.py` docstrings
- **Cases** (`models.py`: `Case`, `Source`, `Person`, `PersonCase`, `Identifier`, `Message`,
  SQLite via SQLModel in `db.py`) are isolated investigations — each has its own uploaded sources
  and its own person pool. Uploading a chat export (`POST /cases/{id}/sources`, multipart) creates
  a `Source` row and runs ingestion (`ingestion.py`) on a background thread pool, so the request
  returns immediately with `status=pending` and the UI polls until it flips to `ready`/`failed`
- **Person identity is case-scoped, not global-by-name**: within one case, a repeated raw sender
  name resolves to the same `Person`; the *same name in two different cases* is never assumed to
  be the same person — that's Phase 2's reviewed merge flow, not automatic. `Person` rows
  themselves are global (not case-scoped) specifically so a future cross-case merge doesn't
  require restructuring the schema — see `ingestion.py`'s `_find_or_create_person`
- The `DetectiveEngine` / `build_all_profiles()` results are no longer built once at import time —
  `engine_cache.py` builds them lazily per case from that case's ready `Message` rows, and drops
  the cache entry whenever a source in that case finishes ingesting
- `cases.html` is the case dashboard: create a case, upload sources (WhatsApp works; Instagram/
  Telegram are selectable but their parsers raise `NotImplementedError` until Phase 3 — an upload
  fails visibly with a clear status instead of silently doing nothing), watch import status
- `detective_lang.html` and `person.html` are scoped by `?case_id=` in the URL (set by navigating
  from `cases.html`); the old dead `<select>` "case" dropdown is gone
- `server.py` is still a **dev server**, not the production backend — no auth, in-process
  `ThreadPoolExecutor` instead of Celery/Redis, SQLite instead of Postgres+pgvector. That's a
  deliberate Phase 1 choice (see Phase 5 below), not an oversight
- Two demo cases are seeded on first run (`seed_demo_case.py`), each independently checked/created
  by name so adding a new one doesn't require wiping an existing install's database:
  - **"Demo: Study Group"** from `sample_chat.txt` + `sample_dm_riya_karan.txt`
  - **"Demo: Housemates"** (`sample_housemates.txt` + `sample_dm_meera_dev.txt`) - a deliberately
    more complex case for exercising the sentiment/volatility/group-vs-DM features: Meera has a
    genuine day-over-day emotional arc (calm Mon/Tue → frustrated/anxious Wed/Thu over a looming
    thesis deadline → relieved Fri → happy Sat), and is markedly more vulnerable in her DM with Dev
    than she lets on in the group (`context_comparison` fires: originally "Tone runs more negative in
    DMs than in groups" under VADER; with the Hinglish sentiment model it says "Comes across more
    emotionally variable in DMs than in groups" - her DM swings from "scared" (-0.7) to "proud" (+0.6),
    so the average stays near zero but the volatility is 0.65 vs 0.52); Dev is a steady, low-volatility dry-humor baseline; Ishaan is a steady
    high-positivity baseline. Both are ingested through the same real upload path as any user file
  - Message wording in both was calibrated against VADER's actual scores (`vaderSentiment`) when the
    demo was written (the app now scores with the Hinglish model - see "Hinglish sentiment" below; Meera's
    Thursday is -0.40 there vs -0.14 under VADER), not
    just written to *sound* emotional - e.g. "i just feel like i'm not handling this well" scores
    *positive* in VADER because of "well", so it was rewritten to "i feel terrible, like i'm
    completely failing at this". Worth remembering for any future synthetic data: naturalistic
    subtext reads fine to a person but often doesn't move VADER's lexicon-based score at all

As of **Phase 2**, identity resolution is real, not a stub:

- After every successful ingestion, `identity_resolution.scan_for_matches(case_id)` compares that
  case's people against every person in every *other* case across two active evidence tiers -
  **hard** (the raw sender string looks like a phone number and matches identically across cases -
  WhatsApp shows a raw number instead of a saved contact name when the sender isn't in the address
  book) and **fuzzy** (name similarity - both plain edit-distance and token containment, so "Karan"
  vs "Karan Mehta" fires even though edit-distance alone under-scores it for the length
  difference). Whichever tier clears its threshold gets recorded as a `MergeSuggestion` row - never
  applied automatically. A pair that already has a suggestion (any status) is never re-suggested,
  so a rejected call isn't silently re-litigated on the next scan
  - There's a third tier, **soft**, but it's currently **disabled** - see the Improvement Stage
    section below for why. It's not a stub; it was tried twice (embedding-centroid similarity, then
    a stylometric fingerprint) and both proved unreliable at this corpus size, with real numbers to
    show it. Both scores are still computed and shown in `evidence` for transparency even though
    neither gates a suggestion right now.
- `merge_review.html` lists pending suggestions with the evidence behind each one; accepting one
  (`POST /merge-suggestions/{id}/accept`) performs a real merge - the lower-id `Person` survives,
  the other's `Identifier` rows are repointed to it, missing `PersonCase` rows are added, and the
  duplicate `Person` is deleted. Any *other* pending suggestion referencing the removed person is
  repointed to the survivor or dropped if that would self-pair or duplicate another suggestion -
  verified working via a real multi-suggestion cascade in testing, not just the happy path
- Case-scoped views (`/cases/{id}/investigate`, `/cases/{id}/people`) are completely unaffected by
  merges - they still key off raw sender name within that case's own sources, exactly as in Phase 1.
  What changes is `/cases/{id}/person/{name}` now also returns `person_id` and `linked_case_count`,
  and once `linked_case_count > 1`, `person.html` shows a banner linking to `combined_dossier.html`
- `combined_profile.py`'s `build_combined_profile` pools a person's messages across every case
  they've been merged into, reusing `person_profile.py`'s aggregation core
  (`build_person_profile_from_sources`) - the same sentiment/traits/vocabulary math as the
  case-scoped dossier, just fed a differently-assembled `sources` list. This is also why
  `person_profile.py`'s internal filtering changed from "by exact sender name" to "by
  caller-supplied pre-filtered messages": name-based filtering breaks once the same real person has
  different raw names in different cases, so `combined_profile.py` filters by `Identifier.id`
  instead, which is exact regardless of name spelling

As of **Phase 5**, a case has a real workspace (`workspace.html?case_id=`, linked from every case's
page) with Search / Timeline / Evidence tabs, and the pipeline underneath now keeps timestamps:

- **Timestamps** - every parser returns `sent_at` and ingestion stores it in `Message.sent_at` (the
  column existed since Phase 1 but was never filled). WhatsApp is the hard case: an export's date
  order follows the phone's locale, so "12/01/23" is 1 Dec or 12 Jan. `parsers/whatsapp.py` settles
  it from the file where the data can (a day above 12 rules an order out; a chronological export
  rules out an order that would run backwards) and only when it genuinely can't - a short export
  whose every date is <= 12 - reads it month-first (the layout its line regex accepts is WhatsApp's
  en-US style) and **says so**: parsers take an optional `notes` list, ingestion stores the caveats
  on `Source.date_note`, and the UI shows them beside the source. Instagram's `timestamp_ms` is UTC
  while WhatsApp/Telegram are the exporting device's local time, so in a cross-platform case the
  Instagram times are off by the user's UTC offset - flagged in that source's note, not correctable
  without a per-case timezone. A source whose dates can't be read is stored undated, never guessed.
  Sources ingested before Phase 5 are backfilled at startup (`ingestion.backfill_timestamps`) by
  re-parsing the stored file and matching by `seq`, so no Message/Identifier id changes (pins rely on
  that). `db.py` gained a small `_COLUMN_MIGRATIONS` list because `create_all` never adds columns to
  an existing table
- **Search** - `GET /cases/{id}/messages` (logic in `workspace.py`): every word must match,
  `"quoted"` for an exact phrase, case-insensitive, optional sender/source/date filters; with no
  query it is the plain filtered browse the timeline's day view uses. Matching is Python-side
  `re.IGNORECASE`, not SQL LIKE (LIKE is only case-insensitive for ASCII and needs wildcard
  escaping); highlight segments come back pre-split from the server because Python indexes by code
  point and JS by UTF-16 unit, so raw offsets would drift after an emoji. A message opens in context
  (+-5 neighbours in its source): `/cases/{id}/messages/{mid}/context`. Everything is case-scoped:
  another case's message id is a 404
- **Evidence pins** - `Pin` table (one per message, plus a free-text note); routes are case-scoped
  (`/cases/{id}/pins[/{pin_id}]`, GET/POST/PATCH/DELETE) and pinning is idempotent. Pin from search
  results, the timeline's day view, or a message's context rows; the Evidence tab lists pins in the
  order the messages were sent and saves notes when a box loses focus
- **Timeline** - `GET /cases/{id}/timeline`: per day, messages per person and their average sentiment
  compound (the same score the emotion signal uses - it reads wording, not sarcasm or
  subtext, and the page says so). The chart is bars (activity) over a person x day mood grid on a
  validated diverging ramp (blue positive / red negative / gray neutral, equal steps per arm), with
  a per-day tooltip, a table view, and click-through to that day's messages. Silent days between the
  first and last dated day are drawn as empty cells, not skipped. On the Housemates demo Meera's
  Thursday is the one negative cell, matching the arc that case was written around - Wednesday only
  dips mildly, which is what a lexicon scorer gives
- **Ingestion status** - `ingestion.recover_stranded_sources` runs at startup: a server restart used
  to strand any queued or running import at pending/processing forever (the in-process pool loses
  its queue); such sources are now re-queued if the upload file exists, else failed with a clear
  message. `DELETE /cases/{id}/sources/{sid}` removes a *failed* import only (a failed parse writes
  no rows, so nothing references it; ready sources are deliberately not removable - that would have
  to untangle people, merges and pins). `cases.html` shows per-source message count, date span and
  date note, importing/failed counts on the case list, and a "can't reach the server" note while
  polling, and now escapes all user/file-derived text (it used to put labels into innerHTML raw)
- **Deliberately not done**: the Postgres+pgvector / Celery+Redis migration (the roadmap makes it
  conditional on real usage volume, and there is none yet); removing ready sources; semantic
  (embedding) search; a per-case timezone. **The WhatsApp parser's gaps (en-US-only line regex, dropped
  continuation lines, media placeholders counted as speech) were closed on 2026-09-24 - F-01/F-02/F-03,
  see "Parser fidelity" below**
- Tests: `test_timestamps.py` (pytest) pins the date-order inference, the 12-hour clock edges, and
  that the parser reads exactly the lines the old en-US-only matcher did (`MSG_RE` is kept for that)

**Parser fidelity** (2026-09-24, BACKLOG F-01/F-02/F-03) - what real exports do, and the rules chosen:
- **Layouts (F-01):** `parsers/whatsapp.py` reads Android (`25/12/2023, 9:14 pm - Name: text`: 12- or 24-hour, upper/lower-case
  am/pm, narrow no-break space before it, `/` `.` `-` separators, 2/4-digit or year-first years) and iOS
  (`[25/12/23, 21:14:05] Name: text`, seconds, invisible LRM marks, BOM). When a short export can't settle DD/MM vs MM/DD,
  **only the classic en-US layout (upper-case AM/PM with `/`) is read month-first**, as before; every other layout is read
  day-first (an Indian-locale `dd/mm/yyyy, h:mm pm` phone is the likely first real export) and says so in a note. No note
  is raised when both orders read every line identically. Old behaviour is pinned: the 112 pre-existing parser tests pass unchanged.
- **Multi-line messages and system lines (F-02):** a line with no date header continues the previous message; a header with
  no `Name: ` (joined, left, encryption notice) is a system event, ends the current message and is skipped - not kept as a
  row. Message count/order for single-line exports is unchanged, so `seq` and pins are safe. **Old sources keep the (possibly
  truncated) text they were ingested with; re-upload to get full multi-line text.** `backfill_timestamps` still skips a source
  whose re-parse no longer matches its stored text exactly, so an old undated source with multi-line messages is skipped
  with a warning (safe, not half-updated).
- **Media / deleted (F-03):** `Message.kind` = `text` | `media` | `deleted` (`parsers.whatsapp.classify_text`; only a message that
  is *entirely* a placeholder - a caption next to `<Media omitted>` stays text). Rows are kept (same ids/seq); `load_case_sources`,
  `combined_profile` and `identity_resolution` read text only, so engine, profiles, sentiment, graph and eval never see them
  (a photo sent as a reply is not counted as a turn or exchange - a deliberate trade-off for one rule at one choke point).
  The timeline counts every message but scores mood from text only: a media-only/deleted-only person-day has `mood: null` and is
  drawn hatched. Existing installs: the column is added by `_COLUMN_MIGRATIONS` and `ingestion.backfill_message_kinds()`
  re-labels old WhatsApp rows at startup (verified on a copy of the real DB: exactly the 4 pure `<Media omitted>` rows
  changed, 18 captioned ones stayed text). Only WhatsApp is classified; Instagram/Telegram media entries are dropped by their parsers.
- **Tests:** `conftest.py` points every pytest run at a throwaway DB (`DETECTIVE_DATABASE_URL`) so the suite can never touch
  `detective.db`; `test_whatsapp_formats.py` (F-01/F-02) and `test_message_kinds.py` (F-03, through the real ingestion path).
  Each new behaviour was mutation-checked (a deliberate breakage fails exactly the tests aimed at it). Full suite: 332 passed,
  2 xfailed with the model built (after the Phase 7 pilot's tests); a fresh clone without it: 305 passed, 29 skipped.
- **The Paper Leak eval moved after F-03, and it was not an improvement - now fixed (E-1, 2026-09-24):** removing 4 media
  placeholders changed top-1 from 44.1% to 47.2% (model) and 46.9% to 53.5% (VADER fallback), because `eval_attribution.py`
  shuffled every sender's messages with one shared RNG, so a change to one sender's list could re-draw other senders' held-out
  sets (in a synthetic check: 12 of 19 list sizes). `eval_split.py` now gives each sender its own generator seeded from
  (seed, split index, sender) and the eval averages 5 splits; see "Improvement Stage" for the resulting mean +- sd.

**Demo: The Paper Leak** - a fourth seeded case built for the investigation phases (6 and 7 below): nine
people and ~730 messages over Sep 1 - Nov 17 2025 - the class group (480 msgs) plus five DMs, two of them
between *other* people (witnesses handing over their own chats). Yash is warm, generous and over-helpful,
and is gradually narrowed down as the person selling leaked mid-sem papers (he works weekend nights at the
print shop the exam cell used, and sells the papers through a Telegram channel while circulating a "guess
list" as cover). The other eight are written to be distinguishable but not trivially so: the bro-slang trio
(Yash/Harshit/Parth), the two formal time-precise writers (Meenakshi/Nikhil), the two emoji-heavy ones
(Tanvi/Kritika), the two terse ones (Sana/Arjun). Files: `sample_leak_group.txt` and
`sample_leak_dm_{yash_kritika,yash_harshit,yash_parth,meenakshi_sana,arjun_nikhil}.txt`.
`sample_leak_case_key.json` is the **answer key**: 16 planted contradictions (C01-C16, each with the
suspect's statements and what they conflict with, severity, and type - self-contradiction, contradicted by a
witness/photo/gate register, shifting story, knowledge slip, group-vs-DM), 6 items of corroborating conduct,
and 5 innocent **decoys** (a self-corrected memory slip, a witness whose statement changes for an
understandable reason, a buyer's contradiction, a real chai seller also named Lalit, a procedural
inconsistency) that a detector should *not* score as the suspect lying. Every reference is an exact
timestamp + sender + quote, and `test_leak_case.py` checks each one really exists in the chats (and that every
line of every sample parses - the parser silently drops non-matching lines). Dates are settled by the data
(no ambiguity note). Baseline on this case, untouched: attribution top-1 accuracy 49.7% when first recorded;
**latest measurement 2026-09-24 (after F-03 and E-1, mean +- sd over 5 splits, 142 test messages each): 51.7% +- 3.3 with the
VADER fallback, 49.6% +- 2.1 with the Hinglish model** (chance is 11%). Older single-split figures for this case (44.1%-53.5%)
were split noise (BACKLOG E-1, now fixed). Coverage is 0.0% - the engine never commits on this case, because the "uncertain" thresholds were set
for 3-person cases and nine speakers dilute the softmax. That is honest behaviour and is more evidence for
Improvement Stage item (1), fitting weights/thresholds; it is recorded, not tuned. Ingesting this case
exposed a real inefficiency: `identity_resolution.score_pair` re-embedded every person once per *pair*
(162 embedding passes for 81 pairs, ~30s per scan, 255s to seed); `scan_for_matches` now memoizes each
person's centroid/fingerprint per scan (results bit-identical, seed 255s -> ~42s)

**Investigation board rework** (after first real use on the 9-person case, where the Phase 4 board was a
238x239px clump: it ran a live force simulation with link attraction and released a node the moment you let go,
so everything sprang back together). `investigation_board.html` was rewritten:
- **Layout is placed, not simulated.** A deterministic one-time layout (strong repulsion, label-sized
  collision, weak frequency-based links, stretched to the board's aspect ratio) spreads people out; after
  that nothing moves except what you drag. Positions are saved per case (`BoardLayout` table,
  `GET/PUT/DELETE /cases/{id}/board-layout`, validated: finite numbers, bounded count and size) and restored
  on reload; Auto-arrange deletes the saved arrangement and lays out again. Pan, zoom, fit, full screen, and
  keyboard (Enter selects, arrows move the focused person).
- **Lasers.** Straight beams with a crisp bright core and a blurred colored glow, additive where they cross,
  coloured by tone: blue warm / red tense / gray neutral - the same blue/red as the timeline's mood scale, so
  colour means the same thing everywhere (swap `--warm-glow` in the page's CSS if amber reads better as warm).
  The blur filter uses `userSpaceOnUse` because SVG's default bounding-box region erases a perfectly
  horizontal line. Thickness = number of exchanges.
- **Selection glow.** A selected person gets an amber ring and a pulsing blurred halo; their beams brighten
  and everything else fades. Hovering does the same temporarily. (Nodes are raised only once they actually
  move: raising on mouse-down makes the browser drop the click.)
- **Filters** (which double as the legend): tone chips with counts, a minimum-exchanges slider, "only links of
  selected", reset. People with no visible link are dimmed. Tone is decided server-side in
  `graph_analysis.tone_for` so the page and the relationship panel can't disagree: warm > +0.15, tense
  < -0.05 (mean sentiment compound of the replies exchanged; set under VADER, re-checked for the Hinglish model), and **unclear when there are fewer than 5 exchanges** -
  a pair with two replies is never called "tense" from one unlucky message.
- **Readable labels.** Opaque name plates placed by a small search (below/above/side/diagonal of each person,
  least collision with other plates and other people, three passes), re-run after every move; full metadata
  lives in an HTML side panel (a person's strongest, warmest and tensest links, or the two-or-more-person
  relationship inference), so nothing depends on cramped canvas text. Verified on the 9-person case with
  people dragged on top of each other: 0 plate/plate and 0 plate/other-person collisions.
- **A data fix found on the way:** exchanges were counted over all sources flattened into one list, so the
  last message of one chat and the first of the next counted as a reply (exactly one spurious exchange per
  source boundary); `graph_analysis` now walks each source separately. The attribution engine's
  `transition_counts` has the same flaw (BACKLOG F-18).
Before -> after on the 9-person case (1600x950 window): board 766x480 -> 1212x758 (larger still in full
screen), people spread 238x239 -> 882x552px, closest pair 24 -> 165px.

**Hinglish sentiment** (the app's sentiment source is now `hinglish_sentiment.py`, not plain VADER). Measured
first, on real data: on the SemEval-2020 SentiMix Hindi-English tweets VADER gets **49% accuracy / 47% macro-F1**,
finds no sentiment word at all in ~44% of texts and catches only **24% of negative messages**; on Hinglish
YouTube comments it is *below the majority-class guess*. Plain Hinglish sentences it scores exactly 0.000
("bahut acha laga", "yeh bilkul acha nahi hai", "bakwaas hai sab"). Everything that reads sentiment was blind
to Hinglish negativity: the board's tone filters and inference, the timeline mood grid, the dossier, and the
attribution engine's emotion signal. What was built:
- **The model.** Character + word n-gram TF-IDF (copes with Romanized Hindi's free spelling), VADER's own scores
  as extra inputs (so English keeps VADER's knowledge), and four features from `hinglish_lexicon.py` - a hand-curated
  Hinglish emotion lexicon (~230 words/phrases with spelling variants, strength, intensifiers, and Hindi-style
  negation where the negator comes *after* the word, "acha nahi") and a curated list of unambiguous emoji - fed to
  a logistic regression trained on Hindi-English tweets + Hinglish YouTube comments + English tweets + ~236
  chat-register lines (`sample_hinglish_chat_train.tsv`). Output is deliberately VADER-shaped:
  `polarity_scores(text)` -> neg/neu/pos/compound with compound = P(pos) - P(neg), so no caller had to change;
  `polarity_scores_batch` / `emotion_features_batch` are the fast path (~4,800 messages/s; a 50k-message chat in
  ~10s). `explain(text)` says which n-grams and lexicon words pushed the score. 13 MB model file, offline at run time.
- **Emoji are treated as register-specific** (found by error analysis on chat text, then fixed in two steps). Emoji
  that flip with context (the crying face, skull, pleading face, sweat-smile, folded hands ...) get *no polarity* - the
  words decide. Every emoji is stripped from the text the n-grams and VADER see, and only a curated list of unambiguous
  ones (laughing, love, celebration; anger, sadness) reaches the model as features - **and those features are learned
  from the chat-register rows only, zeroed for tweets/comments/English**, because in tweets a laughing emoji often
  marks mockery. Before this a bare laughing reaction scored -0.61 and "go get it!! fire" -0.44 (VADER itself reads
  the fire emoji as negative); now +0.97 and +0.62, and acknowledgement emoji (thumbs-up, OK-hand) are not a polarity.
  Letter stretching ("yessss") is squashed.
- **Safe failure.** If the model file is missing, or was built with different features (`FEATURE_VERSION`
  mismatch), the app logs a warning and falls back to plain VADER instead of running mismatched.
- **Evaluation** (`eval_sentiment.py`, `sentiment_data.py`; `--retrain` downloads the public datasets into the
  git-ignored `.sentiment_cache/`, retrains, evaluates, writes `sentiment_metrics.json`). Held-out results, VADER -> model,
  macro-F1: SentiMix test **46.9 -> 68.7** (negative recall 24% -> 79%);
  English tweets (regression check) **65.5 -> 72.5**; blind-labeled chat messages, test half,
  **53.9 -> 64.0** (accuracy 56.1 -> 69.4); Hinglish YouTube comments
  40.6 -> 62.2 (**a real cost**: the earlier model that read emoji the tweet way scored 67.5 there - the trade
  was made for chat). On 60 fresh plain-Hinglish sentences written *before* the lexicon existed
  (`sample_hinglish_probes.json`, `probes2`): **VADER 47% -> model 83%**.
What did not work, kept as findings: a model trained on SentiMix alone hit 68% in-domain but **0% negative recall on
sentences from another domain** and wrecked English (macro-F1 42%) - it learns a domain, not "Hinglish"; only the
multi-domain pool with English data and VADER features generalized. Blending VADER with the model (any weight) traded
Hinglish accuracy for chat accuracy and beat neither. Embedding features added ~1 point at 200x the cost, so weren't kept.
Existing Hinglish models on Hugging Face were checked and rejected: the "Hinglish" RoBERTa is actually trained on English
tweets, and the XLM-R one on the same YouTube data with self-reported F1 0.67.
Effects on the rest of the app, measured with paired comparisons over 5 splits x 3 cases (910 predictions): attribution
accuracy VADER 48.6% vs model 47.7% (flips 55 vs 47, not significant - a single-split "drop" of 5 points was noise), so the
attribution engine is unaffected within error. Also added, additively: Hinglish words for the topic families, question /
command / connector / pronoun detection and dossier stop-words (`HINGLISH_*` sets in `detective.py`, `HINGLISH_STOPWORDS` in
`person_profile.py`) - **no measurable effect on the English-heavy demo chats and no measured harm, so they are unproven
coverage, not an improvement**.
**Honest limits:** (1) **the chat gain is probably overstated**: I wrote the chat-register training lines and labeled
the chat evaluation set, with the same conventions, on synthetic chats - the real test is your own chats, and
`eval_sentiment.py` will take labeled real messages (BACKLOG F-25); (2) on the English-heavy demo chats the model still
**catches fewer negatives than VADER** (47% vs 63% recall, out of only 19 test negatives); (3) known misses on plain
Hinglish (2 sentences are marked `xfail` in `test_hinglish_sentiment.py`; on the unseen probes 10 of 60 are wrong, e.g. "bhai tu
toh chha gaya"); (4) Devanagari script and sarcasm are not handled; (5) training data licences: SentiMix OpenRAIL, YouTube
CC-BY-4.0, the English tweets' licence is unstated - fine for local use, check before redistributing the model file.

**Improvement Stage** (in progress) - before more features, measuring and fixing accuracy problems
found through real testing rather than guessing:

- `eval_attribution.py` is a real eval harness for the attribution engine: per case, holds out ~20%
  of each sender's messages, builds a `DetectiveEngine` on the rest, and reports top-1 accuracy,
  coverage (% not flagged uncertain), precision-when-confident, and a per-sender breakdown. This
  didn't exist before - "accuracy" was pure eyeballing until now. Since E-1 (2026-09-24) it averages
  **5 reproducible splits** (`--splits N`, 1 = quick) and reports mean +- sd; each sender's split
  depends only on that sender (`eval_split.py`, tested by `test_eval_split.py`, whose independence
  tests fail if the old shared-generator design comes back).
  **Baseline** (hand-tuned weights, untouched; mean +- sd over 5 splits, VADER fallback / Hinglish model):
  Demo: Study Group (18 test msgs/split, chance 33%) - top-1 44.4% +- 13.6 / 40.0% +- 9.1, coverage 43% / 40%,
  precision when confident 51.3% / 50.0%. Demo: Housemates (21/split) - 61.0% +- 8.5 / 59.0% +- 8.7, coverage
  45% / 44%, precision 68.1% / 71.7%. **On Study Group the engine is not clearly distinguishable from chance**
  (within ~1 sd), and the model vs fallback gaps (2-4 points) are all inside one sd. The old single-split
  figures (44.4% / 38.9% / 42.9%, 57.1% / 47.6% / 60.0%) were one draw each. Precision
  when confident is modest among 3 people (33% chance) - real evidence the hand-picked signal
  weights and uncertainty thresholds need fitting against labeled data, not more guessing (deferred
  as its own item below - see why)
- `eval_identity.py` is a self-contained regression fixture for identity resolution's soft tier,
  built from the permanent sample files (not transient test cases, so it's rerunnable forever):
  true positives are one real person's own messages split in half; true negatives are any two
  different real people. Used it to test two different candidate soft-tier signals:
  - Embedding-centroid cosine similarity (the original signal): true negatives scored 0.56-0.84,
    true positives 0.69-0.79 - completely overlapping, not separable
  - A stylometric fingerprint (function-word frequency rates + `stylometric_features`, a classic
    authorship-attribution technique - added specifically to fix this): still not separable. A true
    negative ("Karan" vs "Aman") scored 0.913, *higher* than every true positive (max 0.851)
  - Root cause looks like data volume, not signal choice: these synthetic people have ~200-400
    words total each, and standard stylometric techniques are validated on thousands of words per
    author. So the soft tier is **disabled** in `identity_resolution.py`'s `score_pair` (not
    threshold-tuned to fit this one 6-person fixture, which would just be overfitting) until real
    usage provides meaningfully more messages per person - re-run `eval_identity.py` before ever
    re-enabling it
- Deferred, in order: (1) fit the six attribution signal weights against labeled held-out data via
  `eval_attribution.py`, once there's enough real data to do that without overfitting a handful of
  people; (2) replace `syntactic_features()`'s hand-rolled word-list heuristics with a real POS
  tagger (spaCy) - the one place a genuinely new dependency looks like a clear win, worth confirming
  against the eval harness rather than assuming

## Roadmap

**Phase 1 — De-hardcode + case foundation.** ✅ Done (see "Current state" above). SQLite-backed
Case/Source/Person/Message model, upload-driven import dashboard, parser registry, per-case scoped
engine/profiles. Deliberately no cross-case identity logic yet.

**Phase 2 — Identity resolution + cross-case linking.** ✅ Done (see "Current state" above).
hard/fuzzy matching pipeline (soft tier implemented but disabled - see Improvement Stage),
`merge_suggestion` table + human review UI (`merge_review.html`), real person merges on accept,
combined cross-case dossier (`combined_dossier.html`). The original "hard-match (phone/username)"
plan was narrowed to "hard-match (phone-shaped sender string)" since Phase 1's WhatsApp parser only
captures a raw name/number string, not a separate structured identifier field - true structured
hard-match (platform user IDs) arrives with Phase 3.

**Phase 3 — More ingestion sources.** ✅ Done. Real `parse_instagram`/`parse_telegram` in
`parsers/instagram.py` / `parsers/telegram.py`, replacing the `NotImplementedError` stubs. Instagram
(Meta's "Download Your Information" JSON): reverses the export's newest-first message order to
chronological, fixes Meta's well-known mojibake bug (non-ASCII text is UTF-8 bytes wrongly decoded
as Latin-1 - fixed via `text.encode("latin1").decode("utf8")`), skips unsent/no-content entries.
Telegram (Telegram Desktop's "Export chat history" JSON): already chronological, skips non-`type:
"message"` entries (joins/pins/etc), flattens the `text` field's mixed string/entity-object list
(formatted text like links) into plain text, skips entries with no sender. Both raise a clear
`ValueError` if the file doesn't have the expected `"messages"` key, same as the "no messages
parsed" check the WhatsApp path already had - a bad upload fails visibly (`status=failed`), never
silently. A third seeded demo case, **"Demo: Other Platforms"** (`sample_instagram_karan_zoya.json`
+ `sample_telegram_group.json`), deliberately reuses "Karan" and "Riya" from "Demo: Study Group" -
ingesting it produces two real cross-*platform* `MergeSuggestion`s automatically (verified: fuzzy,
100% confidence, both), the first time the project's original "recognize the same person across
different platforms" claim has actually been demonstrated rather than just stated. Revisit
Celery+Redis if the in-process thread pool stops being enough for real import volume.

**Phase 4 — Relationship graph ("investigation board").** ✅ Done. `investigation_board.html` -
D3.js (first external JS dependency - CDN, `d3@7`) force-directed graph, corkboard-styled (pin-style
nodes sized by message count, curved "string" edges). `graph_analysis.py`'s `build_graph`/
`infer_relationship` derive edges entirely from signals the engine already computes: interaction
count from the same adjacent-turn structure `DetectiveEngine.build()` walks for
`transition_counts` (just keeping the exchanged text instead of only a count), topic overlap from
`category_profiles` cosine similarity, sentiment-between from `emotion_features` on exactly the
adjacent-exchange texts. An edge/inference with fewer than 2 shared exchanges reports `uncertain:
true` and every derived field `null` instead of guessing - same discipline as the attribution
engine, verified live (a pair with zero shared exchanges correctly renders "not enough shared
interaction to say anything meaningful yet", not a fabricated read). Click a node to select it;
2+ selected shows the relationship-inference read, 1 selected links to its dossier. **Case-scoped
only** for this pass, not the combined cross-case view the original roadmap line mentioned -
deliberately deferred (see the file's own comment) since it adds real complexity (cross-case edge
computation, node dedup) on top of an already-substantial first version.

**Phase 5 — Case workspace polish.** ✅ Done, except the infrastructure move (see "Current state"
above). Evidence pinning, timeline view, free-text search across pooled messages, ingestion-status
polish - plus the timestamp capture they depended on. Still open, on purpose: migrate SQLite →
Postgres+pgvector and thread pool → Celery+Redis **if real usage volume justifies it** — the schema
in `models.py` was written to make that swap a connection-string/column-type change, not a rewrite.

**Phase 6 — Contradiction detection.** Planned, requested, not started; the method is being designed with the owner
(see BACKLOG N-1 for the design so far: local candidate generation and structured checks, **Gemini as the semantic
judge** behind the `judge.py` placeholder - off by default, `NotImplementedError` until built - and Google's free-tier
terms forbidding personal data; **the owner decided (2026-09-24) that Gemini is used on synthetic demo data only and real
chats are never sent to it**, to be enforced by a per-case flag, not by convention). Given a person of interest, surface statements that
conflict: with themselves over time in one chat; across contexts (group vs DM, and across platforms once
cases are linked); with what other people say; with timestamps and records (a message sent while they claim
to be asleep, a gate register, a photo's time); knowledge they shouldn't have (a slip); and an explanation of
the same event that keeps changing. Constraints that follow from the principles above: every finding cites
both statements (source, timestamp, sender) and says *why* they conflict; the system proposes candidate
pairs and a human confirms them (a confirmed one can be pinned as evidence); it never asserts guilt; and it
must be able to say "uncertain" - in particular it should not score decoys as lies (a self-corrected memory
slip, an inconsistency the person later explains, a namesake, a buyer's contradiction). Evaluated against
`sample_leak_case_key.json`: recall on the high-severity contradictions, precision against the decoys.

**Phase 7 — Investigator-style presentation.** In progress (started 2026-09-24). The owner's brief: a futuristic
digital investigation workstation - a cyberpunk investigation terminal crossed with digital-forensics /
intelligence-analysis software; dark command-center atmosphere; high information density without chaos; **not** a generic
SaaS dashboard. Charcoal/near-black surfaces (never pure black), neon as a restrained accent, not decoration; subtle
scanlines, thin technical borders, glowing active states, grid, tiny status indicators, terminal-style labels, monospaced
metadata, grain, data-density indicators, animated connection lines where appropriate; avoid excessive glow or gradients,
giant rounded cards, glassmorphism and the generic AI-dashboard look.
- **Colour contract (a colour means one thing everywhere):** cyan = active / system state; blue = analytical data; amber =
  warnings and UNCERTAIN evidence (also degraded or failed states); **red = suspicious / high-risk ONLY** (never errors, and
  not negative mood or "tense" links); green = verified / confirmed by a person; white/gray = ordinary text. State is never
  carried by colour alone (a word or glyph too). Every indicator in the shell is wired to real state (`GET /status`) - none is
  decorative (same no-fake-logic rule as the rest of the project).
- `static/theme.css` is the single design system: tokens, panels with bracket corners, mono `//` labels, tags, segmented
  meters, dense tables, tabs, module tiles, the evidence strip, and the grid / scanline / grain layers (with
  `prefers-reduced-motion`). `test_theme_contrast.py` fails if any text colour falls below WCAG contrast on any surface, if a
  `-dim` accent is used as text, or if red is used outside an explicit `.red` state. `static/shell.js` builds the workstation
  bar: brand, case crumb, case-scoped nav, and live indicators (API, NLP scorer, external-LLM switch, stored messages, pending
  identity-review badge). Fonts: Chakra Petch (display), IBM Plex Sans (body), IBM Plex Mono (data), from Google Fonts.
- **Done:** design system + shell + `/status` + `cases.html` as the pilot (checked in a headless browser at 1440 and 390 px,
  and the whole create / import / fail / remove flow re-tested against a real server).
- **To do, one page at a time:** `workspace.html` (the timeline's mood scale and the hatched "no text" cell must move to the
  contract - use the dataviz skill), `investigation_board.html` ("tense" links must stop being red; animate the links),
  `detective_lang.html`, `person.html`, `combined_dossier.html`, `merge_review.html` (accepted = green, pending = amber).
  Pages not migrated yet still carry their own old CSS and no bar. The old noir direction (Special Elite, amber, corkboard) is
  superseded.

## Core design principles (keep these across the codebase)
- Never silently auto-merge identities — across sources within a case, or across cases — always
  create a `merge_suggestion` row and require human confirmation (Phase 2)
- Every attribution result must be able to say "uncertain" rather than forcing a top pick when
  confidence is genuinely low
- Keep person/identifier/message separated — a person can have many identifiers across platforms,
  never collapse this
- Behavioral profiles are context-split (group vs DM) — don't average a person's behavior into one
  flat global profile
- Prefer explainability: every confidence score should be able to show *why* (which
  stylometric/emotional "tells" contributed), not just a number
- Cases are isolated by default — a case's people/sources/engine never leak into another case
  unless a human explicitly links them (Phase 2)
- No placeholder/fake logic in a real code path: an unsupported/malformed import raises a clear
  error and surfaces as a visible `failed` status, never a silent empty result. (Was Instagram/
  Telegram via `NotImplementedError` through Phase 2; both are real parsers as of Phase 3 - the
  same principle now applies to a genuinely wrong file upload, via each parser's `ValueError`)

## Stack
Python (FastAPI, sentence-transformers, scikit-learn), SQLite via SQLModel (Postgres+pgvector once
Phase 5 needs it), in-process `ThreadPoolExecutor` for ingestion (Celery+Redis once Phase 5 needs
it), the existing HTML/JS frontend (or React later, if the noir visual direction survives the port).
D3.js (`investigation_board.html`, via CDN) is the one external JS library in the frontend so far -
force-directed graph layout is a well-solved problem, not worth hand-rolling.

## Reference files in this repo
- `detective.py` — attribution engine (6-signal scoring, softmax ranking, dynamic trust-based
  weight redistribution); re-exports `parse_whatsapp` from `parsers/whatsapp.py` for backward
  compat; `embed_texts` is the public accessor other modules (identity_resolution.py) reuse
- `person_profile.py` — case-scoped person-dossier aggregation (sentiment, Big Five estimate,
  vocabulary, group-vs-DM comparison); `build_person_profile_from_sources` is the shared core also
  used by `combined_profile.py` for the cross-case view
- `combined_profile.py` — cross-case person dossier: pools a merged person's messages across every
  case they belong to, via `Identifier.id` rather than name matching
- `identity_resolution.py` — Phase 2 hard/fuzzy matching pipeline (`scan_for_matches`,
  `score_pair`) that produces `MergeSuggestion` rows; never merges anything itself. Soft tier
  (`stylometric_fingerprint`, `person_centroid`) computed for evidence but disabled from gating a
  suggestion - see Improvement Stage notes above
- `eval_attribution.py` — eval harness for the attribution engine: per-case train/test splits (5 by default),
  reports accuracy/coverage/precision as mean +- sd. Run after any change to signals/weights/thresholds
- `eval_split.py` — its light (no ML/DB imports) per-sender split + `summarize`; `test_eval_split.py` pins it
- `eval_identity.py` — self-contained regression fixture for identity resolution's soft tier,
  built from the permanent sample files. Run before ever re-enabling the soft tier
- `graph_analysis.py` — Phase 4 relationship-graph computation (`build_graph`,
  `infer_relationship`), entirely derived from signals the engine already computes; case-scoped. Also owns
  the tone rules (`tone_for`) and counts exchanges per source
- `models.py` — Case/Source/Person/PersonCase/Identifier/Message/MergeSuggestion/Pin/BoardLayout SQLModel tables
  (Phase 5: `Message.sent_at` is now filled, `Source.date_note` and `Pin` are new; F-03: `Message.kind` text/media/deleted)
- `db.py` — SQLite engine/session setup; `_COLUMN_MIGRATIONS` adds columns introduced after a DB was
  first created (`create_all` only creates missing tables)
- `parsers/` — parser registry, all three implemented: `whatsapp.py` (Phase 1), `instagram.py` /
  `telegram.py` (Phase 3 - see the Roadmap entry above for each format's quirks). Contract (Phase 5):
  `parser(path, notes=None) -> [{sender, text, sent_at, kind?}]`; caveats about how dates were read go to `notes`;
  `kind` (F-03, WhatsApp only today) defaults to "text"
- `ingestion.py` — background ingestion of an uploaded source into Identifier/Person/Message rows;
  case-scoped exact-name person resolution; triggers `identity_resolution.scan_for_matches` after a
  successful ingest (best-effort - a matching failure doesn't undo the ingestion). Phase 5: stores
  timestamps + `Source.date_note`; `backfill_timestamps()` and `recover_stranded_sources()` run at startup
- `engine_cache.py` — lazy per-case `DetectiveEngine`/profile/timeline building + cache invalidation
- `seed_demo_case.py` — seeds all four demo cases from the sample files through the real ingestion
  path; each checked/created independently by name so a new demo case can be added without wiping
  an existing install's database
- `server.py` — FastAPI dev server: case CRUD, source upload/ingestion status, case-scoped
  `/investigate`/`/people`/`/person/{name}`, merge-suggestion list/accept/reject, combined
  `/people/{person_id}`, `/cases/{id}/graph` + `/cases/{id}/relationship-inference`, and (Phase 5)
  `/cases/{id}/senders`, `/messages` (search/browse), `/messages/{mid}/context`, `/timeline`, `/pins`,
  `DELETE /cases/{id}/sources/{sid}` (failed imports only), `/board-layout` — still not the production
  backend (no auth, no Celery/Redis; CORS is wide open - BACKLOG S-2)
- `ui_static.py` — static-file policy for the UI: serves only top-level `*.html` pages and files directly inside `static/`
  with a UI extension (`.css .js .svg .woff2`). The server used to mount the whole project directory, which exposed
  `detective.db`, `uploads/`, the source and `.git/` (fixed 2026-09-24, BACKLOG S-1); `test_ui_static.py` pins it (light:
  no ML imports). It also pins the MIME types, since Windows can serve `.js` as `text/plain`
- `static/theme.css`, `static/shell.js` — the Phase 7 design system and the workstation bar (see Phase 7 above);
  `status.py` — `GET /status` (sentiment scorer really loaded, external-LLM switch, counts) for the bar's indicators;
  `test_status.py`, `test_theme_contrast.py`
- `judge.py` — Phase 6 placeholder (BACKLOG N-1): `Judge` interface, provisional `Statement`/`StatementPair`/`Verdict`
  types and `GeminiJudge`, which checks its config (`DETECTIVE_ALLOW_EXTERNAL_LLM=1`, `GEMINI_API_KEY`,
  `DETECTIVE_GEMINI_MODEL`) and then raises `NotImplementedError` - it sends nothing and never invents a verdict.
  Light (no google-genai or ML import); `test_judge.py` pins the off-by-default behaviour
- `sample_chat.txt` / `sample_dm_riya_karan.txt` — synthetic WhatsApp exports for "Demo: Study Group"
- `sample_housemates.txt` / `sample_dm_meera_dev.txt` — synthetic WhatsApp exports for "Demo:
  Housemates", the emotionally-varied case (day-over-day arc + group-vs-DM contrast)
- `sample_instagram_karan_zoya.json` / `sample_telegram_group.json` — synthetic Instagram/Telegram
  exports for "Demo: Other Platforms" (Phase 3); deliberately reuse "Karan"/"Riya" from "Demo:
  Study Group" so ingesting them demonstrates real cross-platform identity resolution
- `cases.html` — case dashboard: create a case, upload sources, watch import status (Phase 5: counts,
  date span, date notes, remove failed imports), link to Identity Review and the workspace
- `workspace.py` — Phase 5 query logic: message search/browse, context, timeline aggregation, evidence
  pins; server.py holds the thin routes (same split as `graph_analysis.py`)
- `workspace.html` — case workspace, scoped to `?case_id=`: Search / Timeline / Evidence tabs
- `test_timestamps.py` — pytest regression for parser timestamp capture and WhatsApp date-order inference
- `hinglish_sentiment.py` — the app's sentiment scorer (VADER-shaped output, batch path, `explain`, fallback);
  `hinglish_lexicon.py` — its curated Hinglish emotion lexicon; `sentiment_data.py` — dataset download + deterministic
  splits; `eval_sentiment.py` — evaluation / `--retrain`; `hinglish_sentiment_model.joblib` (13 MB) and
  `sentiment_metrics.json` — the trained model and its measured results; `sample_hinglish_chat_train.tsv`,
  `sample_hinglish_chat_labels.json`, `sample_hinglish_probes.json` — its chat-register training lines, the blind-labeled
  chat set (dev/test) and the plain-Hinglish probe sets; tests: `test_hinglish_sentiment.py`, `test_hinglish_language.py`
- `test_graph_analysis.py` — pytest for the board's data rules (per-source exchanges, when a tone may be asserted)
- `BACKLOG.md` — the task bucket list: agreed next features, fine-tunings for unverified/rough areas, ideas.
  Keep it current: add an item when a limitation or unverified area turns up, tick it when fixed
- `sample_leak_group.txt` / `sample_leak_dm_*.txt` — synthetic WhatsApp exports for "Demo: The Paper Leak"
  (nine people, ~730 messages); `sample_leak_case_key.json` — its machine-readable answer key (planted
  contradictions, corroborating conduct, decoys); `test_leak_case.py` — checks the key against the chats
- `detective_lang.html` — query-driven investigate UX, scoped to `?case_id=`
- `person.html` — per-person dossier page, scoped to `?case_id=`; shows a banner linking to the
  combined dossier once a person has been merged across cases
- `merge_review.html` — pending cross-case match review: evidence + accept/reject
- `combined_dossier.html` — cross-case dossier for a merged person, scoped to `?person_id=`
- `investigation_board.html` — relationship board (D3.js), scoped to `?case_id=`: draggable people with saved
  positions, laser-beam links, tone/exchange filters, glowing selection; see "Investigation board rework" above
