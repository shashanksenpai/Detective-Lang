# Detective Lang

**Who said this?** Detective Lang is a speaker-attribution and chat-analysis workbench. Give it chat exports from WhatsApp, Instagram or Telegram and it can:

- rank who in a chat most likely wrote a given sentence, with a confidence score and a per-signal breakdown of *why*, and say **"uncertain"** when the evidence is thin;
- recognize that the same person appears in different chats and platforms, by *suggesting* matches for a human to accept or reject;
- build a per-person dossier (tone, vocabulary, writing habits, group-vs-DM behaviour);
- draw a relationship board of who talks to whom and in what tone;
- search, browse and timeline a case's messages, and pin messages as evidence.

It is a working prototype that runs entirely on your own machine. It understands English and Romanized Hindi ("Hinglish"). It is not a forensic instrument: every result is a probabilistic estimate, and the [measured accuracy](#measured-results-and-limitations) is modest and reported honestly below.

> **Status:** research prototype, single-user, local only. There is no authentication, so read the [security notes](#security-and-responsible-use) before running it with real chats.
> All chats bundled in this repository are **synthetic**. Nothing here comes from real people.

## Contents

1. [Capabilities](#capabilities)
2. [Design principles](#design-principles)
3. [How it works](#how-it-works)
   - [Architecture at a glance](#architecture-at-a-glance) · [Data model](#data-model) · [Ingestion and parsers](#ingestion-and-parsers) · [The attribution engine](#the-attribution-engine) · [Hinglish sentiment](#hinglish-sentiment) · [Person dossiers](#person-dossiers) · [Identity resolution](#identity-resolution) · [Relationship board](#relationship-board) · [Case workspace](#case-workspace)
4. [Demo cases](#demo-cases)
5. [Measured results and limitations](#measured-results-and-limitations)
6. [Getting started](#getting-started)
7. [HTTP API](#http-api)
8. [Repository layout](#repository-layout)
9. [Status and roadmap](#status-and-roadmap)
10. [Security and responsible use](#security-and-responsible-use)
11. [Data, models and licences](#data-models-and-licences)

---

## Capabilities

| Capability | What you get | Page |
|---|---|---|
| **Cases** | Isolated investigations. Each case has its own uploaded sources and its own pool of people; nothing leaks between cases unless you explicitly link people. | `cases.html` |
| **Multi-platform import** | WhatsApp `.txt`, Instagram JSON ("Download Your Information") and Telegram Desktop JSON exports. Imports run in the background with a visible `pending → processing → ready / failed` status. A bad file fails loudly; it never yields a silent empty result. | `cases.html` |
| **"Who said this?"** | Paste a sentence and get a ranked list of candidate speakers with a calibrated confidence for each, a "could be" set when it can't commit, and the evidence behind the ranking. The default **learned engine** learns each person's wording from their own messages and says "uncertain" when its own held-out checks say it can't be trusted; the original six-signal engine is one click away. | `detective_lang.html` |
| **Honest uncertainty** | If the top candidate is below 35% confidence, or ahead of the runner-up by less than 8 points, the result is flagged `uncertain` instead of forcing a pick. | `detective_lang.html` |
| **Person dossier** | Sentiment level and volatility, distinctive words and phrases, question/command rates, topic mix, a heuristic Big Five *writing-style* estimate, and a **group-vs-DM comparison** that surfaces context-dependent behaviour. | `person.html` |
| **Cross-case identity resolution** | The same person under a phone number in one chat and a name in another, or "Karan" vs "Karan Mehta", is *suggested* as a match with the evidence shown. Nothing merges until you accept it. | `merge_review.html` |
| **Combined dossier** | After a merge, one profile pooled across every case and platform the person appears in. | `combined_dossier.html` |
| **Relationship board** | Draggable people, laser-beam links coloured by tone (warm / neutral / tense), sized by number of exchanges. Filters, pan/zoom, saved layout, and a relationship read for any two or more people. | `investigation_board.html` |
| **Workspace** | Full-text search with phrase matching and filters, a message-in-context view, a per-day timeline of activity and mood, and evidence pins with notes. | `workspace.html` |
| **Hinglish-aware sentiment** | A small offline model that reads Romanized Hindi and English, replacing an English-only lexicon that was nearly blind to Hinglish negativity. | used everywhere sentiment appears |

## Design principles

These are deliberate constraints that shape the code, not accidents.

- **Never auto-merge identities.** A match becomes a `MergeSuggestion` row awaiting a human decision. A pair that was suggested once, in any status, is never re-suggested, so a rejected call is not re-litigated.
- **Every attribution can say "uncertain".** Forcing a top pick when the evidence is weak is a failure mode, not a feature.
- **Explainability over a bare number.** Every score can show which signals contributed. The scorer can also say which words pushed a sentiment reading.
- **Person, identifier and message stay separate.** A person can have many identifiers across platforms; they are never collapsed into one row.
- **Behaviour is context-split.** Group and DM behaviour are profiled separately and then compared, never averaged into one flat profile.
- **Cases are isolated by default.**
- **No demographic inference.** The tool does not guess gender, age or similar attributes from writing style. The Big Five output is described as a writing-style summary, explicitly *not* a clinical or diagnostic assessment.
- **No fake logic in a real code path.** Unsupported or malformed input raises a clear error that surfaces as a visible `failed` status.
- **Measure before tuning.** Accuracy claims come from the eval scripts in this repo, and several things that *didn't* work are kept as recorded findings (see below).

---

## How it works

### Architecture at a glance

```
 chat export (.txt / .json)
        │  POST /cases/{id}/sources
        ▼
 parser registry ──── whatsapp │ instagram │ telegram
        │   [{sender, text, sent_at}]  +  caveats about how dates were read
        ▼
 ingestion (background thread pool) ───────────►  SQLite (SQLModel)
        │                                          Case · Source · Person · Identifier · Message
        │ on success                               MergeSuggestion · Pin · BoardLayout
        ▼                                                   ▲
 identity_resolution.scan_for_matches                       │ read per case
   └─► MergeSuggestion (status = pending)             engine_cache (lazy, per case,
                                                       dropped when a source changes)
                                                          ├─ DetectiveEngine   (who said it?)
                                                          ├─ person profiles   (dossiers)
                                                          ├─ graph_analysis    (board)
                                                          └─ workspace         (search/timeline/pins)
                                                                  │
                                                                  ▼
                                              FastAPI (server.py) ──► static HTML/JS pages
```

Stack: Python, FastAPI, SQLModel/SQLite, sentence-transformers, scikit-learn, VADER. The frontend is plain HTML/CSS/JS with one shared design system (`static/theme.css`) and a workstation bar (`static/shell.js`) whose indicators read live state from `GET /status`; D3 v7 (loaded from a CDN) is its only external library and is used by the relationship board. All seven pages use the dark forensic-workstation look.

### Data model

Defined in [`models.py`](models.py).

| Table | Purpose |
|---|---|
| `Case` | One isolated investigation. |
| `Source` | One uploaded export: platform (`whatsapp`/`instagram`/`telegram`), context (`group`/`dm`), import `status`, `error_message`, and a `date_note` recording any caveat about how timestamps were read. |
| `Person` | A real-world person. **Global, not case-scoped**, so a later cross-case merge does not require restructuring the schema. |
| `PersonCase` | Which cases a person belongs to. |
| `Identifier` | One raw sender string (a name or a phone number) within one source, pointing at a `Person`. |
| `Message` | Text, `sent_at` (nullable), `seq` (its position in the source) and `kind`: `text` (something a person wrote), `media` (a photo/video/sticker placeholder) or `deleted` (a deleted-message notice). Non-text rows stay in the record but are not analysed as speech. |
| `MergeSuggestion` | A candidate cross-case match: tier (`hard`/`fuzzy`/`soft`), confidence, JSON evidence, and `pending`/`accepted`/`rejected` status. |
| `Pin` | One evidence pin per message, with a free-text note. |
| `BoardLayout` | Where you dragged each person on a case's relationship board. |

Person identity is **case-scoped on ingest**: within one case a repeated sender name resolves to one `Person`, but the *same name in two different cases* is never assumed to be the same person. That link only ever comes from a reviewed merge.

### Ingestion and parsers

Parsers live in [`parsers/`](parsers/) behind a registry (`PARSERS`). The contract is `parser(path, notes=None) -> [{sender, text, sent_at, kind?}]` in chronological order (`kind` defaults to `text`). Anything the parser had to assume is appended to `notes` and stored on the `Source`, so the UI shows the assumption instead of hiding it.

- **WhatsApp** (`parsers/whatsapp.py`). The hard problem is the date: an export's day/month order follows the *phone's locale*, so `12/01/23` is either 1 December or 12 January and nothing on the line says which. The parser settles it from the file itself where the data allows: a day above 12 rules an order out, and a chronological export rules out an order that would run backwards. Only when a short export is genuinely ambiguous (every date ≤ 12) does it fall back to a default and **say so** in the source's date note: month-first for the classic en-US layout (upper-case AM/PM with `/`), day-first for every other layout. A file whose dates can't be read at all is stored undated, never guessed.

  It reads the layouts real phones write: Android with 12-hour (upper- or lower-case am/pm, including the narrow no-break space newer exports put before it) or 24-hour clocks, `/`, `.` or `-` separators, 2- or 4-digit or year-first years, and iOS's `[date, time:seconds] Name: text` with its invisible marks and byte-order mark. A line with no date header continues the previous message, so **multi-line messages are joined**; a header with no sender (someone joined, the encryption notice) is a system event and is skipped. Messages that are *entirely* a placeholder (`<Media omitted>`, `<attached: …>`, `image omitted`, `This message was deleted`) are stored with `kind` `media` or `deleted`, so they stay visible but are excluded from profiles, sentiment, embeddings, turn-taking and the relationship graph. These layouts were checked against synthetic and hand-written files in each format, **not against a real phone export**.
- **Instagram** (`parsers/instagram.py`). Meta exports messages newest-first, so they are reversed to chronological order. It also repairs Meta's well-known encoding bug (UTF-8 bytes mis-decoded as Latin-1, which corrupts emoji and non-ASCII text). Instagram times are UTC while WhatsApp/Telegram are device-local; that mismatch is flagged in the source note rather than silently "corrected".
- **Telegram** (`parsers/telegram.py`). Skips non-message entries (joins, pins, …), flattens the mixed string/entity list Telegram uses for formatted text, and skips entries with no sender or no text.

Ingestion (`ingestion.py`) runs on an in-process `ThreadPoolExecutor`, so the upload request returns immediately with `status=pending` and the UI polls. On success it invalidates that case's cached engines and runs the identity scan (best effort; a matching failure does not undo a good import). At startup the server creates tables and runs a small column migration, seeds the demo cases, **backfills timestamps** for older sources by re-parsing the stored file and matching by `seq` (so no message id changes and pins survive), and **recovers stranded imports** (a restart used to leave queued jobs stuck at `pending` forever).

### The attribution engine

Two engines answer "who said this?". The **learned engine** is the default; the original **six-signal engine** stays selectable (the *Engine* toggle on the investigate page, `engine` in the API request, or `DETECTIVE_ENGINE=legacy`). The relationship board and the dossiers do not depend on the choice.

#### The learned engine (default)

[`learned_engine.py`](learned_engine.py) learns each person's texting behaviour from their own messages instead of blending hand-weighted similarities. It exists because the six-signal engine could not name Yash for *"Dont worry guys I'll pay"* (his habitual reassure-and-pay line) on the Paper Leak case: its topic-family signal had no word for "pay" or "worry" and switched itself off, the tone and style signals cannot tell three bro-slang writers apart, and it never commits with nine speakers.

- **Features.** Word 1-2-grams and character 2-5-grams (TF-IDF), so a person's wording, spelling habits, slang and emoji all count and "dont" and "don't" share most of their letter patterns. No embedding model is needed.
- **Model.** One multinomial logistic regression per case, retrained when a source is imported, with every person weighted equally so a talkative person does not win ties by volume.
- **Honest confidence.** Probabilities are calibrated on cross-validated predictions of *this case's own messages* (each is predicted by a model that never saw it). The engine says **uncertain** below the confidence at which those held-out checks were right at least **80%** of the time, judged at the pessimistic end of what that many checks can show (a Wilson lower bound), never below 50%, and it will not commit at all with fewer than about 250 checks to learn from. Every answer also carries a **"could be" set** (the fewest people covering 90% of the probability) and the calibration it rests on ("when it was at least 57% sure it was right 82% of the time, and it was that sure for 72% of messages").
- **Exact evidence.** A score is a sum of per-feature contributions, so the explanation is not a story. For the top match over the runner-up, each word (its own weight plus the spelling patterns inside it) and each two-word phrase gets a signed log-odds contribution; together with the model's starting lean they add up exactly to the odds shown, and anything too small to list is folded into "everything else" (`test_learned_engine.py` checks the sum).
- **What it does not do yet.** It reads only the sentence: the messages just before it are accepted and reported as unused. A person's group voice does not fully carry into their DMs, and its confidence describes messages *like those in the chat*. See [Measured results](#measured-results-and-limitations).

A real response from the Paper Leak demo (trained on all 725 messages; nine candidates):

```
sentence: "Dont worry guys I'll pay"       engine: learned      uncertain: false
  Yash 94.7%   Parth 4.4%   Kritika 0.4%   Tanvi 0.3%   ...
why Yash over Parth (log-odds):  worry +1.8  guys +1.7  i'll +0.6  dont +0.1  starting lean +0.9
                                 "pay" -1.9 (points to Parth)  "i'll pay" -0.2      total +3.1  (22 : 1)
calibration: commits at >= 57% confidence; held-out checks there were right 82% of the time, for 72% of messages
```

The six-signal engine on the same sentence: *uncertain*, Parth 16.6%, Yash 13.9%.

#### The original six-signal engine

[`detective.py`](detective.py) scores every candidate speaker on **six signals**, blends them, and converts the blend to confidences.

For each sender the engine builds a profile from that sender's messages. For a query sentence it compares the sentence against each profile:

| Signal | What it captures | How it's computed | Base weight |
|---|---|---|---|
| **topic** | What the message is about, in meaning rather than words | Cosine similarity between the message's sentence embedding (`all-MiniLM-L6-v2`, 384-d) and the sender's mean embedding. "I'm freaking out about the test" lands near "worried about our exam" despite sharing almost no words. | 0.25 |
| **category** | Which topic *family* it belongs to | Cosine similarity between the message's distribution over five curated lexicons (studies, fun, work, logistics, food, plus Hinglish members of each) and the sender's typical distribution. | 0.25 |
| **discourse** | Conversational turn-taking | Given who spoke just before, how often does this sender reply to that person? Laplace-smoothed transition counts, so an unseen pair isn't a hard zero. | 0.15 |
| **emotion** | Emotional baseline | `1 / (1 + mean absolute difference)` between the message's sentiment fingerprint (compound/neg/pos) and the sender's average. A chronically anxious speaker scores high on a new worried message even with zero shared words. | 0.13 |
| **style** | Surface habits | Same similarity form over message length, emoji count, `!` count, `?` count and capitals ratio. | 0.12 |
| **syntactic** | Sentence *form*, independent of topic | Same similarity form over question / exclamatory / imperative / declarative flags, clause density and first-vs-second-person ratio (rule-based, English + Hinglish word lists). | 0.10 |

**Context.** If you pass the messages immediately before the sentence, the topic and category signals use a recency-weighted blend of the sentence and its context (decay 0.6 per step back), so a near-featureless reply like "crazy right?" inherits the topical grounding of what was just said, and the turn-taking signal switches on.

**Trust-based weight redistribution.** A signal that has nothing real to say for a given query should not dilute the result with noise. The `category` signal goes quiet when no lexicon word matched, and `discourse` goes quiet when no context was supplied. Their weight is redistributed proportionally to the signals that are always well-defined, keeping the weights summing to 1. The response reports the `weights_used`.

**Confidence.** Combined scores go through a softmax with a sharpening factor of 8. The query is flagged **uncertain** if the top confidence is under 35% or the lead over second place is under 8 points.

A real response from a fresh checkout (no model built, so plain VADER; case *Demo: Study Group*, no context):

```
sentence:  "the deadline is tomorrow and I haven't started"
uncertain: true      dominant_category: studies      sentence_type: declarative
weights_used: topic .312 · style .150 · emotion .163 · category .250 · syntactic .125 · discourse 0.0

  Aman    43.2%   topic .487  style .967  emotion .903  category .731  syntactic .897  discourse 0
  Karan   36.8%   topic .431  style .989  emotion .892  category .706  syntactic .917  discourse 0
  Riya    20.0%   topic .426  style .898  emotion .885  category .475  syntactic .898  discourse 0
```

The top pick leads by only 6.4 points, under the 8-point margin, so the engine declines to commit. Note how `discourse` was zeroed and its weight moved to the other signals.

**What is not fitted.** The six weights and both thresholds are hand-chosen (the learned engine above is what replaced fitting them). The numbers below show why that mattered.

### Hinglish sentiment

Sentiment feeds the emotion signal, the dossier, the board's tone, and the timeline's mood grid. The app used to run plain VADER, an English lexicon. On Hindi-English text that was close to blind: measured on the SemEval-2020 SentiMix Hindi-English tweets it scored **49% accuracy / 46.9% macro-F1**, found no sentiment word at all in about 44% of texts and caught only **24% of negative messages**. Plain sentences like "bahut acha laga" or "bakwaas hai sab" scored exactly 0.

[`hinglish_sentiment.py`](hinglish_sentiment.py) replaces it with a small offline model:

- **Features:** character (2–5) and word (1–2) n-gram TF-IDF, which copes with Romanized Hindi's free spelling (`acha/accha/achha`, `nahi/nhi`); VADER's own scores as extra inputs, so English keeps VADER's knowledge; and four features from a hand-curated Hinglish emotion lexicon ([`hinglish_lexicon.py`](hinglish_lexicon.py)) with intensifiers and Hindi-style negation, where the negator comes *after* the word ("acha nahi").
- **Model:** logistic regression, trained on a multi-domain pool (Hindi-English tweets, Hinglish YouTube comments, English tweets, and ~236 chat-register lines written for this project).
- **Output is VADER-shaped:** `polarity_scores(text)` → `neg/neu/pos/compound`, with `compound = P(pos) − P(neg)`, so no caller had to change. `explain(text)` reports which n-grams and lexicon words pushed the score.
- **Emoji are treated as register-specific.** Emoji whose meaning flips with context (😭 💀 🥺 😅 🙏 …) get *no polarity*, so the words decide. Every emoji is stripped from the text the n-gram model and VADER see; a small curated list of unambiguous ones (laughing, love, anger, sadness) reaches the model as features, and those features are learned **only from the chat-register rows**, because in tweets a laughing emoji often marks mockery. Before this fix a bare laughing reaction scored −0.61.
- **Safe failure.** If the model file is missing, or was built with a different feature version, the app logs a warning and falls back to plain VADER instead of running mismatched.

**Measured on held-out data** (`python eval_sentiment.py`; VADER → model; **reproduced from a fresh clone with `--retrain`**):

| Held-out set | n | Macro-F1: VADER → model | Notes |
|---|---:|---|---|
| SentiMix test (Hindi-English tweets) | 3000 | 46.9 → **68.7** | negative recall 24% → 79% |
| Hinglish YouTube comments | 957 | 40.6 → **62.2** | a real cost: a model that read emoji the tweet way scored 67.5 here; that was traded for chat accuracy |
| English tweets (regression check) | 2000 | 65.5 → **72.5** | Hinglish support must not cost English |
| Blind-labeled chat messages, test half | 157 | 53.9 → **64.0** | see the caveat below |

**Findings that did not work** (kept, not hidden): a model trained on SentiMix alone hit 68% in-domain but **0% negative recall on sentences from another domain** and wrecked English; it learns a domain, not "Hinglish". Only the multi-domain pool with English data and VADER features generalized. Blending VADER with the model at any weight beat neither. Sentence-embedding features added about one point at ~200× the cost, so weren't kept. Existing Hinglish models on Hugging Face were checked and rejected.

**Honest limits:** the chat-set gain is probably overstated, because the chat-register training lines and the chat evaluation set were written and labeled by the same person on synthetic chats (the real test is your own chats; `eval_sentiment.py` accepts a labeled set). On English-heavy demo chats the model catches *fewer* negatives than VADER (47% vs 63% recall, on only 19 test negatives). Devanagari script and sarcasm are not handled. The model does **not** ship in this repository; see [Getting started](#getting-started).

### Person dossiers

[`person_profile.py`](person_profile.py) pools a person's messages across every source in a case, **split by context (group vs DM)**, and computes:

- **Sentiment:** mean compound, *volatility* (population standard deviation of the per-message compound), and % positive / negative / neutral.
- **Writing habits:** average message length, emoji and exclamation rates, question rate, imperative rate, self-reference ratio, topic distribution, and topic diversity (normalized Shannon entropy across the five topic families).
- **Vocabulary:** top words, most frequent bigrams, and **distinctive words**, meaning words this person uses at least twice whose rate is far above their share of the whole corpus (a keyness ratio, not raw frequency). English and Hinglish function words are filtered so the lists show what someone talks about, not "hai / tha / nahi".
- **Big Five (OCEAN) estimate:** a transparent heuristic mapping from those measurements (for example, extraversion from emoji + exclamation frequency and fun-topic share; neuroticism from emotional volatility and the rate of negative-toned messages). Each score ships with its `basis` string and a disclaimer: it is a **writing-style summary, not an assessment of the person**.
- **Group-vs-DM comparison:** plain-language insights when the gap between contexts crosses a threshold (e.g. "Comes across more emotionally variable in DMs than in groups").

The combined dossier (`combined_profile.py`) reuses the same aggregation core but pools by `Identifier.id` rather than by name, which stays exact even when one person has different raw names in different cases.

### Identity resolution

[`identity_resolution.py`](identity_resolution.py) runs after every successful import, comparing that case's people against everyone in every *other* case across evidence tiers:

| Tier | Rule | Confidence | Status |
|---|---|---|---|
| **hard** | The raw sender string looks like a phone number (≥ 7 digits) and matches identically after normalization. WhatsApp shows a raw number when a contact isn't saved, so this is strong. | 0.97 | active |
| **fuzzy** | Name similarity ≥ 0.82, taking the better of an edit-distance ratio and *token containment*, so "Karan" vs "Karan Mehta" fires even though edit distance alone under-scores the length difference. | the similarity | active |
| **soft** | Embedding-centroid and stylometric-fingerprint similarity. | n/a | **disabled** |

Every match is recorded as a `MergeSuggestion` with its full evidence (names compared, every tier's raw score, even sub-threshold ones) and shown in `merge_review.html`. **Accepting** one performs a real merge: the lower-id person survives, the other's identifiers are repointed to it, missing case memberships are added, the duplicate person is deleted, and any *other* pending suggestion referencing the removed person is repointed to the survivor or dropped (if it would self-pair or duplicate). Case-scoped views (`/investigate`, `/people`, the case dossier) are unaffected by merges.

**Why the soft tier is off.** It was tried twice, first with embedding-centroid similarity and then with a classic authorship fingerprint (function-word rates plus stylometric features), and both failed the repo's own regression fixture (`eval_identity.py`: one real person's messages split in half vs. two different real people). Reproduced:

| Signal | Same person (split in half) | Different people | Separable? |
|---|---|---|---|
| Embedding centroid | 0.689 – 0.788 | 0.557 – 0.840 | no |
| Stylometric fingerprint | 0.723 – 0.851 | 0.586 – **0.913** | no |

A pair of *different* people ("Karan" vs "Aman") scored 0.913, higher than every same-person pair. No threshold fixes overlapping ranges, and picking one to fit a six-person fixture would just be overfitting. The likely root cause is data volume: classic stylometry is validated on thousands of words per author, and these people have a few hundred. Both scores are still computed and shown in the evidence for transparency, but neither gates a suggestion. Re-run `eval_identity.py` before ever re-enabling the tier.

### Relationship board

[`graph_analysis.py`](graph_analysis.py) derives everything from signals the engine already computes; there is no separate hand-authored metric.

- An **exchange** is two consecutive messages by different senders *within the same source*. (Counting across all sources flattened into one list used to create one spurious exchange per source boundary; this was found on the 9-person demo and fixed.)
- **Topic overlap** is the cosine similarity of the two people's topic-family profiles. **Sentiment between** is the mean compound of the replies exchanged.
- **Tone:** `warm` above +0.15, `tense` below −0.05, otherwise `neutral`, and **`unclear` whenever there are fewer than 5 exchanges**, so a pair with two replies is never called "tense" from one unlucky message. An edge with fewer than 2 exchanges is marked `uncertain` and every derived field is `null`.
- **Relationship read** for a selection of two or more people: dominant shared topic → `professional` (work / studies / logistics) or `personal` (fun / food), with the same uncertainty rule.

The page ([`investigation_board.html`](investigation_board.html)) uses a **placed, not simulated** layout: an earlier version ran a live force simulation and everything sprang back into a 238 × 239 px clump the moment you released a node. Now a deterministic one-time layout spreads people out (repulsion, label-sized collision, weak links, stretched to the board's aspect ratio) and after that only what you drag moves. Positions are saved per case (`BoardLayout`), and *Auto-arrange* discards them. On the 9-person demo the board went from 766 × 480 to 1212 × 758 px and the closest pair of people from 24 to 165 px apart. Beams are coloured by tone on the same blue/red scale as the timeline's mood grid, so colour means the same thing everywhere. Selecting a person glows their links and fades the rest, and name plates are placed by a small collision-avoiding search. Tone and minimum-exchange filters double as the legend.

### Case workspace

[`workspace.py`](workspace.py) holds the query logic and `server.py` the thin routes.

- **Search.** Every word must match; `"quoted"` matches an exact phrase; case-insensitive; optional sender / source / date filters. Matching is Python-side `re.IGNORECASE` rather than SQL `LIKE`, since `LIKE` is only case-insensitive for ASCII. Highlight segments are pre-split on the server because Python indexes by code point and JavaScript by UTF-16 unit, so raw offsets would drift after an emoji. With no query it is a plain filtered browse. A message opens in context (± 5 neighbours in its source). Everything is case-scoped: another case's message id is a 404.
- **Timeline.** Per day, messages per person and their average sentiment, drawn as activity bars over a person × day mood grid (blue positive / red negative / gray neutral, equal steps per arm), with a table view and click-through to that day's messages. Silent days between the first and last dated day are drawn as empty cells, not skipped. Every message counts towards the activity numbers (so they match the day view), but mood is scored from text only: a person whose only messages that day were media or deleted notices gets a hatched cell, not a neutral one. The page says plainly that it reads wording, not sarcasm or subtext.
- **Evidence pins.** One pin per message plus a free-text note; pinning is idempotent; notes save when the box loses focus; the Evidence tab lists pins in the order the messages were sent.

---

## Demo cases

Four cases are seeded on first run **through the real ingestion path**, not a shortcut. Each is created independently by name, so a new demo can be added without wiping an existing database.

| Case | Contents | What it demonstrates |
|---|---|---|
| **Demo: Study Group** | A 3-person WhatsApp group + one DM | Baseline attribution and dossiers |
| **Demo: Housemates** | 3-person group + one DM | A day-by-day emotional arc for one person (calm → anxious → relieved → happy) and a group-vs-DM contrast: she is markedly more vulnerable with one housemate than in the group, and the comparison insight fires |
| **Demo: Other Platforms** | An Instagram DM + a Telegram group that reuse "Karan" and "Riya" | Cross-**platform** identity resolution: ingesting it produces two fuzzy merge suggestions automatically |
| **Demo: The Paper Leak** | 9 people, ~730 messages, 6 sources (a class group + five DMs, two between *other* people, handed over as witnesses) | A fictional investigation: a generous classmate is gradually narrowed down as the seller of leaked exam papers. Ships with `sample_leak_case_key.json`, an answer key of 16 planted contradictions, 6 items of corroborating conduct and 5 innocent decoys, each with an exact timestamp, sender and quote, which `test_leak_case.py` verifies against the chats |

The Paper Leak case is the stress test. Its nine speakers are written to be distinguishable but not trivially so (three bro-slang writers, two formal time-precise ones, two emoji-heavy ones, two terse ones).

---

## Measured results and limitations

Reproduce the attribution numbers with `python eval_attribution.py --engine both`. For each case it holds out ~20% of every sender's messages, builds each engine on the rest, and reports top-1 accuracy (forced choice), top-3, log-loss (how badly a wrong answer is believed), the "could be" set, **coverage** (how often the engine was willing to commit rather than say "uncertain") and **precision when it does commit**, plus a **paired** count of the messages only one engine got right. It repeats this over **5 reproducible splits** (`--splits 1` is a quick single run) and reports **mean ± standard deviation**, because on a small case a single split can swing by several points. Each sender's split depends only on that sender, so unrelated changes to the data don't re-draw everyone else's test messages. `--transfer` trains on the group chat only and tests on the DMs; `--curve` caps the training messages per person.

| Case | Senders | Test msgs / split | Chance | Legacy top-1 | **Learned top-1** | Top-3 (legacy → learned) | Log-loss (legacy → learned) | Only legacy right / only learned right | Learned commits |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| Study Group | 3 | 18 | 33% | 40.0% ± 9.1 | **61.1% ± 8.8** | 100% → 100% | 1.04 → 0.96 | 8 / 27 | never (too little chat) |
| Housemates | 3 | 21 | 33% | 59.0% ± 8.7 | **67.6% ± 7.8** | 100% → 100% | 0.94 → 0.82 | 18 / 27 | never (too little chat) |
| The Paper Leak | 9 | 142 | 11% | 49.6% ± 2.1 | **73.5% ± 2.6** | 83.7% → 96.1% | 1.82 → 0.79 | 60 / 230 | 71% of messages, right **82.2%** (416/506) |

(Legacy figures are with the Hinglish sentiment model; with the plain-VADER fallback they are 44.4 ± 13.6, 61.0 ± 8.5 and 51.7 ± 3.3. The legacy engine commits on 40%, 44% and 0% of messages, with 50%, 72% and no precision to report.)

**How to read this honestly:**

- **The learned engine is better on all three cases, clearly on two.** On the nine-person Paper Leak it goes from about 50% to about 74% top-1 (and 96% top-3), and of the 710 messages judged (142 × 5 splits) it alone got 230 right against 60 for the legacy engine. On Study Group the gain is about two standard deviations. On Housemates the gain (8.6 points) is under one standard deviation, though the paired count still leans to the learned engine, so treat it as "no worse, probably better". Its "could be" list is 2.7 of 9 people wide and holds the true sender 95.5% of the time; the legacy engine's 90% set is 7.9 of 9 people wide.
- **It commits only when it has earned it.** On Paper Leak it commits on 71% of messages and is right 82.2% of the time, against an 80% target set from its own held-out checks. On the two three-person chats (about 100 messages each) it never commits: below about 250 messages to check itself against, the calibrated cut-off proved unreliable (on the Paper Leak, capping the training size gave 74% precision at 180 messages and 64% at 90, against the 80% target), so it ranks and shows probabilities but says "uncertain". That is a design choice, not a bug, and the 250 was measured on one case.
- **It learns quickly.** Paper Leak top-1 by messages per person in training: 5 → 46%, 10 → 56%, 20 → 60%, 40 → 68%, 80 → 73% (chance 11%).
- **A person's group voice only partly carries into their DMs.** Trained on the Paper Leak group chat and tested on the 249 DM messages: 55.8% top-1 (legacy 35.7%; always guessing the most frequent DM sender 29.7%). Its confidence does not transfer: it commits on 55% of them but is right only 69%, under its 80% target, because the calibration describes messages like those in the chat. By person it ranges from Meenakshi 19/22 and Yash 56/74 to Nikhil 1/19 and Sana 6/25. (In the app the model is trained on the DMs too; this is a stress test.) On the tiny Study Group the same test (31 DMs) scores 45.2%, below always guessing the majority (51.6%).
- **The legacy engine's thresholds (35% / 8 pts) were set for 3-person chats.** With nine speakers the softmax spreads out and it declines everything: it commits on none of the 142 test messages, with either scorer. That is honest behaviour, but it is what the learned engine's per-case calibration replaces.
- **Why the spread is reported.** An earlier version of the eval used a single split and shuffled every sender's messages with one shared random generator, so changing one sender's list could re-draw other senders' held-out messages. Dropping four media placeholders moved the Paper Leak figure from 44% to 47% (model) and from 47% to 54% (fallback) with no change to the engine. That is fixed (`E-1` in [`BACKLOG.md`](BACKLOG.md)), and the standard deviations above show how much a single split can wander.
- **Everything here is on synthetic chats**, written by one author in formulaic personas, with held-out messages drawn at random from the same conversations, so near-duplicates leak and absolute accuracy is optimistic. The *comparison* between the engines and the calibration logic are what should transfer; nothing has been validated on real conversations (`A-5`).
- **What did not work** (kept as findings): raw sentence embeddings appended to the n-gram features lowered accuracy on two of the three cases (Study Group 61.1 → 52.2, Paper Leak 71.0 → 65.9), zero-shot "behaviour prototype" features added nothing, and normalising apostrophes and capitals in the word features cost a little on the small cases. Unweighted classes were worse than equally weighted ones on all three.

**Known limitations**, tracked in [`BACKLOG.md`](BACKLOG.md):

- **Attribution.** The learned engine ignores the messages just before the query (`A-2`), its confidence is weaker across contexts (`A-3`), small chats never commit and "how sure" is fixed at 80% (`A-4`), and a message from someone who is not in the chat is forced onto somebody unless it is uncertain (`A-8`).
- **Import formats.** The WhatsApp layouts above have not been checked against a real phone export, only synthetic and hand-written files, so an unusual locale can still fail with "No messages parsed". System events (someone joined, the encryption notice) are dropped rather than stored. Sources imported before multi-line support keep their old truncated text until re-uploaded. Only WhatsApp media is flagged; the Instagram and Telegram parsers drop media-only entries, and have only been exercised on hand-written samples, never a real export.
- **Timestamps.** Instagram is UTC while WhatsApp/Telegram are device-local, so a cross-platform case is off by the UTC offset (flagged, not corrected). One unreadable date line undates a whole WhatsApp file.
- **Sentiment.** Reads wording only: no sarcasm, no Devanagari script, and a cold accusation with no negative words reads neutral, so the board's "tense" tone can't see it. Known misses on plain Hinglish remain (two are marked `xfail` in the tests).
- **Board.** An "exchange" has no time limit (a reply next morning counts like one seconds later); group and DM exchanges are pooled; it is case-scoped only and has been checked with 3 and 9 people, not 20+.
- **Scale.** Search scans in Python: fine at chat-export size, not at 100k+ messages. No way yet to remove a *ready* source or delete a case.
- **Infrastructure.** SQLite and an in-process thread pool, by design for now. Moving to Postgres + pgvector and Celery + Redis is conditional on real usage.

---

## Getting started

**Requirements:** Python 3.11+ (developed on 3.11), and internet access on first run to fetch the sentence-embedding model (`all-MiniLM-L6-v2`, ~90 MB, cached by Hugging Face afterwards).

```bash
git clone https://github.com/shashanksenpai/Detective-Lang.git
cd Detective-Lang
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Build the sentiment model (recommended)

The trained model file is **not committed** (see [licences](#data-models-and-licences)), so build it once. It downloads about 9 MB of public datasets into a git-ignored `.sentiment_cache/`, trains in about two minutes, evaluates, and writes `hinglish_sentiment_model.joblib` and `sentiment_metrics.json`:

```bash
python eval_sentiment.py --retrain
```

You can skip this step and the app will still run: without the model it logs a warning and falls back to plain VADER (English only, and weak on Hinglish). Training is deterministic, and a fresh retrain reproduced the numbers in this README.

### 2. Run the server

Run from the repository root; the server resolves the database, uploads and sample files relative to it.

```bash
python -m uvicorn server:app --port 8000
```

Then open **http://127.0.0.1:8000/cases.html** (the bare `/` returns 404; there is no index page).

**The first start is slow (about a minute or more):** it loads the embedding model and seeds the four demo cases synchronously before the server answers. Later starts are quick. `uvicorn --reload` is known to hang on the author's Windows setup, so restart manually.

### 3. Try it

Pick a case on the dashboard; each case offers **Open investigate**, **Open dossiers**, **Open investigation board** and **Open workspace**. **Identity Review** is a link at the top of the dashboard.

1. Open **Demo: The Paper Leak** → *Open investigate* and paste `Dont worry guys I'll pay`: the learned engine names Yash and shows the words behind it. Click **Legacy** to re-run the same sentence on the six-signal engine and see it decline. Try a bare `ok` to see the "could be" list, and open **Demo: Housemates** to see a small chat where the learned engine ranks but will not commit. The optional "what was said just before" box takes one `Sender: text` line per prior message, oldest first (the legacy engine uses it; the learned engine says it did not).
2. From the same case, *Open dossiers* and pick Meera to see the group-vs-DM comparison.
3. Open **Identity Review** to see the two suggested cross-platform matches from **Demo: Other Platforms** (Karan and Riya); accept one and open the combined dossier.
4. Open **Demo: The Paper Leak** → *Open investigation board*, then *Open workspace* to search, view the timeline, and pin messages.

To analyse your own chats, create a case and upload an export. In WhatsApp use *Export chat → Without media* (produces a `.txt`); in Instagram use *Download your information → Messages → JSON*; in Telegram Desktop use *Export chat history → JSON*. Read the [privacy note](#security-and-responsible-use) first.

### Tests and evaluation

```bash
pip install pytest
python -m pytest -q          # 420 passed, 2 xfailed once the model is built (about a minute)
python eval_attribution.py   # attribution accuracy / coverage / precision per case, mean ± sd over 5 splits (--splits 1 = quick)
                             #   --engine learned|legacy|both  --transfer (group -> DM)  --curve (messages per person)
python eval_sentiment.py     # sentiment model vs VADER on the held-out sets
python eval_identity.py      # regression fixture for the (disabled) soft identity tier
```

**Without the model file** the 29 model-dependent tests in `test_hinglish_sentiment.py` are skipped, each with the instruction to run step 1, and the other 393 pass. The suite runs against a throwaway database (`conftest.py`), never your `detective.db`. A model file that exists but was built with different features fails rather than skips. The two `xfail`s are known Hinglish misses recorded on purpose.

---

## HTTP API

Interactive docs are available at `/docs` while the server runs. All case data is scoped under `/cases/{id}`.

| Method | Route | Purpose |
|---|---|---|
| `POST` / `GET` | `/cases`, `/cases/{id}` | Create / list cases, get one |
| `POST` | `/cases/{id}/sources` | Upload an export (multipart: `file`, `platform`, `context`, `label`); returns immediately, `status=pending` |
| `GET` / `DELETE` | `/cases/{id}/sources`, `…/sources/{sid}` | List sources with status; remove a *failed* import only |
| `POST` | `/cases/{id}/investigate` | `{sentence, context[], engine?}` (`engine`: `learned` or `legacy`, default from `/status`) → ranked speakers, confidences, `uncertain`, and that engine's evidence (`evidence` / `plausible_senders` / `calibration` for learned, per-signal breakdown for legacy) |
| `GET` | `/cases/{id}/people`, `/cases/{id}/person/{name}` | People in a case; a person's dossier |
| `GET` | `/people/{person_id}` | Combined cross-case dossier for a merged person |
| `GET` | `/status` | Live state for the workstation bar: attribution engine, sentiment scorer, external-LLM switch, counts |
| `POST` | `/cases/{id}/rescan-matches` | Re-run identity scan for a case |
| `GET` | `/merge-suggestions` | Suggestions with evidence (`?status=pending\|accepted\|rejected`) |
| `POST` | `/merge-suggestions/{id}/accept`, `…/reject` | The only place a merge happens |
| `GET` / `POST` | `/cases/{id}/graph`, `/cases/{id}/relationship-inference` | Board nodes/edges; relationship read for 2+ people |
| `GET` / `PUT` / `DELETE` | `/cases/{id}/board-layout` | Saved board positions |
| `GET` | `/cases/{id}/senders`, `…/messages`, `…/messages/{mid}/context`, `…/timeline` | Workspace search, browse, context, timeline |
| `GET` `POST` `PATCH` `DELETE` | `/cases/{id}/pins[/{pin_id}]` | Evidence pins |

---

## Repository layout

```
server.py                 FastAPI dev server: thin routes, serves the UI pages
ui_static.py              Static-file policy: top-level .html pages and static/ assets only
static/                   theme.css (design system), shell.js (workstation bar), dossier.js/css (shared dossier panels)
status.py                 GET /status: real state for the bar's indicators
judge.py                  Placeholder judge interface for contradiction detection (Gemini; off, unimplemented)
models.py  db.py          SQLModel tables; SQLite engine + small column-migration list
parsers/                  Parser registry: whatsapp.py · instagram.py · telegram.py
ingestion.py              Background import, timestamp backfill, stranded-import recovery
engine_cache.py           Lazy per-case engine / profile / timeline cache
engine_kind.py            Which attribution engine answers (DETECTIVE_ENGINE, default learned)
learned_engine.py         The learned attribution engine (default): n-gram model, calibrated abstention, exact evidence
detective.py              The original six-signal attribution engine (still selectable)
person_profile.py         Case dossier aggregation (shared core)
combined_profile.py       Cross-case dossier, pooled by Identifier id
identity_resolution.py    hard / fuzzy (and disabled soft) matching → MergeSuggestion
graph_analysis.py         Relationship graph, tone rules, relationship inference
workspace.py              Search, context, timeline, pins (query logic)
hinglish_sentiment.py     The sentiment scorer (VADER-shaped output, fallback)
hinglish_lexicon.py       Curated Hinglish emotion lexicon + unambiguous-emoji lists
sentiment_data.py         Dataset download + deterministic splits
seed_demo_case.py         Seeds the four demo cases via the real ingestion path

cases.html  detective_lang.html  person.html  combined_dossier.html
merge_review.html  investigation_board.html  workspace.html      The UI pages

eval_attribution.py  eval_sentiment.py  eval_identity.py        Evaluation harnesses
eval_split.py             Per-sender train/test split and mean ± sd helper for eval_attribution.py
eval_metrics.py           Scoring rules for the attribution eval (top-k, log-loss, abstention, paired flips)
test_*.py  conftest.py    pytest suites (parsers, message kinds, graph rules, sentiment, static policy,
                          leak-case key); conftest.py points them at a throwaway database
sample_*.txt / *.json     Synthetic chats, training lines, labeled sets, answer key
sentiment_metrics.json    Last recorded sentiment evaluation
LICENSE                   Apache License 2.0 (this repository's code)
BACKLOG.md                Agreed next work and every known rough edge
CLAUDE.md                 Detailed project notes and design history
```

`BACKLOG.md` and `CLAUDE.md` hold the detailed design history, including the reasoning behind each decision and what was tried and dropped.

---

## Status and roadmap

| Phase | Scope | State |
|---|---|---|
| 1 | Case foundation: SQLite model, upload dashboard, parser registry, per-case engines | Done |
| 2 | Identity resolution, reviewed merges, combined dossier | Done (soft tier disabled, with evidence) |
| 3 | Instagram and Telegram parsers, cross-platform demo | Done |
| 4 | Relationship graph | Done, then reworked into the current board |
| 5 | Workspace: timestamps, search, timeline, pins, import polish | Done (Postgres/Celery move deferred until usage justifies it) |
| Improvement | Evaluation harnesses; Hinglish sentiment; **learned attribution engine with calibrated abstention** | In progress. Done: harnesses, Hinglish sentiment, the learned engine (default). Next: validate on a real chat, use the preceding messages, better cross-context confidence (see `A-2`..`A-8`) |
| 6 | **Contradiction detection**: statements that conflict with themselves, across group and DM, with other people, or with timestamps and records. Every finding must cite both statements, say *why* they conflict, wait for a human to confirm, never assert guilt, and be able to say "uncertain" (decoys such as a self-corrected memory slip must *not* be scored as lies) | Being designed. `judge.py` is a placeholder for an optional, off-by-default Gemini judge (it raises `NotImplementedError` and sends nothing). Will be evaluated against `sample_leak_case_key.json` |
| 7 | Investigator-style presentation: a dark digital-forensics workstation look with a colour contract (red only for suspicious/high-risk) | Done for all seven pages. One item is deliberately left: the blue/red sentiment and tone colours, which still use red for negative (deferred by the owner) |

## Security and responsible use

- **Keep it on localhost; there is no authentication.** Every API route is open to anything that can reach the port, so whoever can would be able to read every case. It binds to `127.0.0.1` by default; do **not** expose it to a network or the internet, or run it with `--host 0.0.0.0`, while it holds real chats. Two specifics. The static file server only serves the top-level `.html` pages and files directly inside `static/`, and the database, uploads, source and `.git` are refused (`test_ui_static.py` covers this). CORS is currently wide open (`allow_origins=["*"]`), so in principle a web page open in the same browser could try to call the local API; browsers restrict that for localhost to varying degrees and it has not been tested here. Restricting origins and checking the `Host` header is tracked as `S-2` in [`BACKLOG.md`](BACKLOG.md) and should land before any shared use.
- **A chat export is untrusted input.** Sender names, source labels and case names come from files you upload, and every page escapes them before display. Four stored cross-site-scripting holes of exactly this kind were found on four pages while building the current UI (a crafted export ran script in the page); each was reproduced in a real browser on the old page and re-tested after the fix. See `S-3` and `S-4` in [`BACKLOG.md`](BACKLOG.md). The learned engine's evidence table also echoes words and phrases from the chat; those go through the same escaping and were checked in a browser with a hostile export. Because the API itself is still open (`S-2`), escaping on output is the only defence, so any new page that shows names or words must be checked with a hostile export.
- **Planned, and not built yet: an optional Gemini judge** for contradiction detection. When it exists it will be off by default and will send the two statements being compared (plus a little context) to Google. Google's terms for the **free** Gemini API tier say submitted content may be used to improve its products and read by human reviewers, and tell you not to submit personal information; the paid tier is not used that way. The maintainer's decision is to use it on the bundled synthetic cases only: real chats will not be sent to it.
- **Your chat content stays local.** Today it is never sent anywhere. The app does contact Hugging Face to fetch and check the embedding model (and, for `--retrain`, to download the public datasets), and the relationship board loads D3 from a CDN, so that page needs internet access.
- **Uploads are stored on disk** under `uploads/` (git-ignored) and in `detective.db` (git-ignored).
- **Treat results as leads, not proof.** Attribution is probabilistic (the learned engine is right about 3 times in 4 on the synthetic nine-person case, and its stated confidence is only as good as the chat it learned from), and the sentiment scorer can't see sarcasm. The tool is built to say "uncertain", to propose rather than decide, and to require a human to confirm merges. It is not designed to assert guilt or to be evidence on its own.
- **Only analyse chats you have the right to analyse**, ideally with the consent of the people in them. The bundled cases, including "The Paper Leak", are entirely fictional.

## Data, models and licences

- **Sentence embeddings:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), downloaded at first run.
- **VADER:** `vaderSentiment`, used as a baseline, as extra inputs to the Hinglish model, and as the fallback scorer.
- **Sentiment training data** (downloaded by `--retrain`, never redistributed here): SentiMix Hindi-English tweets (SemEval-2020 Task 9, OpenRAIL); Hinglish YouTube comments (CC-BY-4.0); a small set of templated Hindi-English sentences (MIT, training only); English tweets (licence not stated by the source). The chat-register training lines, the blind-labeled chat set and the plain-Hinglish probe sets in this repo were written for this project.
- **Why the model file isn't committed:** the mixed licences above make redistributing the derived 13 MB model something to check first, so it is rebuilt locally from the public sources instead.
- **License for this repository's code:** [Apache License 2.0](LICENSE). It covers this repository's code only; the datasets and models above keep their own licences.
