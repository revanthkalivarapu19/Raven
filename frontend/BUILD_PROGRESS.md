# RAVEN Frontend — Progress Log

> Two efforts are recorded here. The **redesign** (current, 2026-09-29) tracks
> `RAVEN_Frontend_Redesign_Phases.md`; the **original rebuild** of the frontend
> (complete, 2026-09-28) tracks its own roadmap and is kept below as an archive.
>
> **Scope, both times: `frontend/` only.** Nothing outside it was touched — verified
> with `git status` after every phase and again in the final audit.

---

# Redesign — 2026-09-29

## Overall Status

**Complete.** Phases 1–12 worked through in order, each verified before the next. No known
blocking issues. No work outside `frontend/`.

## Current Phase

Phase 12 — Dependencies & Notes (**Completed**).

## Completed

| Phase | What it did | Status |
| ----- | ----------- | ------ |
| 1 | Inspected the existing frontend; confirmed build and lint pass before changing anything | Done |
| 2 | Visual direction — refined typography and hierarchy (see below) | Done |
| 3 | Result redesigned around claim → verdict → confidence → explanation → sources | Done |
| 4 | Confidence rebuilt as a radial ring | Done |
| 5 | Sources refined; passage made the dominant element, stance scannable | Done |
| 6 | Explanation given its own labelled section | Done |
| 7 | Follow-up interaction reviewed and preserved | Done |
| 8 | Light/dark theme added, with a switch and persistence | Done |
| 9 | Responsive sweep + interaction polish | Done |
| 10 | Final product polish (dead code, alignment, contrast, overflow) | Done |
| 11 | Full-flow browser testing in both themes | Done |
| 12 | Dependencies (none needed) + documentation | Done |

## What actually changed

The existing frontend was already editorial and already met much of what the roadmap asks
for. Reading the phases against what was built, two requirements were genuinely unmet, and
one instruction asked to reconsider a decision the previous effort had made deliberately:

1. **Confidence was a 2px horizontal bar** (Phase 4 asks for a radial/ring indicator).
2. **There was no dark theme at all.** The previous effort had *deliberately removed* the
   old product's dark mode, and Phase 8 now asks for one back, done properly.
3. **Type was reconsidered** (Phase 2). The Instrument Serif / Inter / JetBrains Mono
   pairing was kept — it already is "distinctive editorial display against clean interface
   sans", which is what the phase asks for — but its *use* changed, which is where the
   weakness actually was.

Everything else was refinement rather than replacement, which is what the roadmap asks
for ("Preserve existing functionality wherever possible").

## Phase log

### Phase 1 — Understand the existing frontend

Read the whole `frontend/` tree: 51 files, the design-token layer, all component
stylesheets, the conversation/result components, both hooks, the service seam and the
internal review page. Ran `npm run build` and `npm run lint` first — both clean — so every
later failure would be attributable to a change rather than to a broken starting point.

**Notable finding:** the roadmap's file (`RAVEN_Frontend_Redesign_Phases.md`) had replaced
the old roadmap in the project root. The repo was otherwise untouched.

### Phase 2 — Visual direction

Kept the typefaces, changed how they are used:

- **The verdict stance moved to the display serif.** "The sources support this claim" is
the sentence the verdict word is short for, so it belongs to the headline, not to the
interface chrome. It was previously UI sans.
- **RAVEN's opening line moved to the display serif** and, more importantly, moved
  *into* the verdict block — it used to sit below the whole result, so "This one is
  fabricated, and there is a paper trail." arrived after the evidence instead of with the
  verdict.
- **The claim as understood is now italic display type**, so a quoted reading never blurs
  into RAVEN's own finding.
- Prose measure tightened from 44rem to 40rem for a more comfortable line length.

### Phase 3 — The result

The result already avoided card nesting, so this was hierarchy work. The block now reads in
the exact order the roadmap lists: claim → verdict → confidence → explanation → sources.
The intro line was moved up to sit under the verdict (above), and the sources legend became
a single hairline row (folio · bar · tally) instead of a labelled grid cell, so it reads as
a legend rather than a widget.

### Phase 4 — Radial confidence

New `components/result/ConfidenceRing.jsx`: an inline SVG ring, 92px, 3.5px stroke, with the
reading as the arc. Design decisions:

- **The arc length is the number.** `stroke-dasharray` is the circumference and
  `stroke-dashoffset` is `circumference × (1 − value)`, so 93% is a nearly closed ring and
  42% is a little under half. Verified numerically: the implied fraction matched the stated
  reading exactly at 0.93, 0.97, 0.42 and 0.08.
- **The ring takes the verdict colour**, so confidence never reads as a free-floating score.
- **The number is real text, not part of the drawing** — selectable and readable by a
  screen reader, with a visually-hidden "Confidence " prefix. The SVG is `aria-hidden`.
- **A zero reading draws no arc at all**, rather than a stray dot from a rounded cap.
- **It draws itself on arrival** (`ring-draw`, 0.62s) — pure CSS, driven by a `--ring-full`
  custom property, so the sweep from empty to the reading is what makes the arc legible as
  a number. `prefers-reduced-motion` collapses it via the existing global rule.
- No dependency was added; it is inline SVG.

The old 2px `.meter` primitive was deleted — it had no remaining consumer.

### Phase 5 — Evidence and sources

The source table already carried all seven fields the roadmap lists (name, type, headline,
date, stance, passage, link). Two changes:

- **The passage became the dominant element** (1rem, 1.66 line-height) because it is what a
  person actually checks the verdict against.
- **The passage's left rule is tinted by stance** (`--verdict-real-line` /
  `--verdict-fake-line`). Enough to scan a column of sources and see where they disagree;
  quiet enough that no row becomes an alert — which is the "restrained" the roadmap asks
  for.

### Phase 6 — The explanation

The reasoning used to be an unlabelled `.turn__prose` div. It is now a `<section>` with a
folio label that names the question it answers:

| Turn kind | Label |
| --------- | ----- |
| assessment | "Why I read it this way" |
| explanation | "The reasoning" |
| evidence | "What the further sources show" |
| answer | "About this source" |

Labels live in `lib/verdict.js`, the single place user-facing wording is written, so
nothing about the internal workflow can leak in with them.

### Phase 7 — Follow-up interaction

Reviewed, not rebuilt. The existing design already does what the phase asks: the newest
answer offers three suggestions at full strength, older ones recede, "Check another claim"
invites rather than submitting, and the composer stays docked and important without
dominating. Re-verified after the redesign that a suggestion chosen further up the page
still answers *its own* claim (see Phase 11).

### Phase 8 — Light and dark theme

This was the largest change. Implemented as **two sets of semantic colour tokens**, not two
stylesheets:

- `:root` holds the light set; `[data-theme='dark']` on `<html>` overrides colours only.
  Type, space, radii, motion and layout are declared once and are theme-independent.
- Every hard-coded colour found during Phase 1 was tokenised first:
  `--ink-hover` (solid-button hover), `--on-accent` (text on the accent), `--scrim` and
  `--scrim-strong` (the About sheet and the lightbox).
- The dark set is designed, not inverted. Warm near-black paper (`#131211`) rather than
  grey; the accent **lifted** to `#8098f0` because a cobalt dark enough to read on white
  is unreadable on black; verdicts re-mixed at lower luminance so they hold their hue on a
dark ground; the "deeper" accent likewise lifted rather than darkened; shadows reduced,
  because depth on dark paper has to come from hairlines.
- `lib/theme.js` + `hooks/useTheme.js` + `components/ThemeToggle.jsx`. The switch is a
  `role="switch"` with `aria-checked`, a sliding knob and a spring easing, and sits in the
  rail — including on small screens, where it is the one rail item that survives
  collapsing.
- **Preference persistence:** first visit follows the operating system; using the switch
  stores an explicit choice in `localStorage` (`raven:theme:v1`), which then wins, including
  over later system changes. `localStorage` — deliberately *not* the `sessionStorage` the
  conversation uses: a preference should outlive the tab and a piece of work should not.
- **No flash on load:** `index.html` carries a small inline boot script that sets
  `data-theme` before the first paint. It duplicates the key and attribute from
  `lib/theme.js`; both say so in comments, because the duplication is the one fragile point
  of this phase.
- `<meta name="theme-color">` is kept in step in both places, so mobile browser chrome
  matches.

### Phase 9 — Responsive and interaction polish

Swept 1920 / 1280 / 834 / 390 in **both** themes and measured rather than eyeballed:

- 0 horizontal overflow at every size; no element off-screen inside the shell.
- Rail collapses to a masthead bar ≤900px with the theme switch intact at a 40px target;
  nav items 40px; composer send/attach 42px; chips and evidence toggles 40px.
- The confidence legend stacks to one column, and the ring row keeps the ring beside the
  reading down to 26rem, below which it stacks.
- Evidence rows move the stance above the source on small screens so the verdict-relevant
  fact is read first, and passages hold 1rem — they are the last thing that should shrink.
- Long content: a 240-character pasted claim takes the `--long` treatment (17px, tighter
  leading) and does not overflow; long source titles do not wrap out of their column.

### Phase 10 — Final product polish

- **Deleted dead code:** the `.meter` primitive (no consumers), the `turn--assessment`
  class (duplicated information already in the child `.assessment`), and the now-unused
  `.evidence__percent` rule.
- **Alignment measured, not guessed:** the ring and the reading are optically centred to
  0px, and the legend folio and its bar centre to 0px.
- **Vertical rhythm is on the 4px scale** throughout the result: 32 / 32 / 24 / 32px between
  the stance, the lead, the confidence row, the legend and the sources.
- Confirmed light and dark produce **identical geometry** — only colour changes, which is
  the check that the theme system is a token layer and not a second layout.

### Phase 11 — Testing

Everything below was observed through headless Chrome over the DevTools Protocol against
the **production build**, driving real events (real `File` objects, real `change` /
`dragover` / `drop`, real key presses, real reloads) and reading the resulting DOM and
computed styles back. Harnesses live outside the project.

- Application starts; build and lint clean.
- Claim input, examples, keyboard send, image attach/preview/remove, non-image rejection,
  window-level drag and drop, image-only send, lightbox open/close — all still work.
- All three verdicts render, with the ring's implied fraction matching the stated reading
  in every case, and the arc colour matching the verdict colour in every case.
- Explanation labels correct per turn kind; 3 reasoning paragraphs on an assessment.
- Multiple sources render with all seven fields and a working external link; the clamped
  passage unfolds (80px → 265px) and re-clamps.
- Follow-ups: an older suggestion still answers its own claim, not the most recent one.
- Both themes, the About sheet, the `#design` page, focus ring, scroll-follow and the
  "Latest answer" jump.
- Responsive layouts and long content at four widths.

### Phase 12 — Dependencies and notes

**No dependency was installed or needed.** The confidence ring is inline SVG, the theme
switch is CSS, and both icons came from the `lucide-react` already in `package.json`. The
redesign instead *removed* a dependency earlier in the project's life (`react-router-dom`,
which the single-surface product never used). `README.md` was updated with the theme layer,
the result composition and the contrast floor; this log records the phase work.

## Issues found and fixed

Real defects found by testing, each fixed and re-verified:

1. **The confidence ring silently lost its verdict colour.** `.ring` declared
   `--verdict-accent: var(--ink-primary)` as a fallback, and a custom property declared on
   the *same element that reads it* overrides the value inherited from `.assessment--real`
   / `--fake` / `--unverified`. The arc and the number rendered in ink instead of the
   verdict colour in every state. Fixed by removing the declaration and putting the
   fallback in the consuming declarations (`var(--verdict-accent, var(--ink-primary))`).
2. **`--ink-tertiary` failed WCAG AA in the light theme** at 3.79:1, and it carries real
   reading text — confidence notes, the evidence accordion, section folios. Darkened to
   `#6e6759` (~5.3:1).
3. **Informative source metadata was using `--ink-quaternary`**, the decorative step —
   source type, the "Already shown" marker, published dates and domains. Promoted to
   `--ink-tertiary`, and `--ink-quaternary` is now reserved for genuinely supplemental
   marks and documented as such.
4. **RAVEN's opening line was in the wrong place**, below the entire result rather than
   with the verdict it qualifies (raised in Phase 2, fixed by rendering it inside the
   verdict block).
5. **`--ink-quaternary` in the dark theme was arbitrarily light** once the light value
   changed; both quiet steps were re-derived from a contrast target rather than by eye.
6. **Two dead selectors** left behind by the redesign (`turn--assessment`, `.meter`).

A seventh candidate — an apparent one-paragraph explanation — turned out to be a bug in the
test probe, not the interface: all three reasoning paragraphs render. Recorded here so the
next reader does not go looking.

## Still needing attention

Nothing blocking. Known and accepted:

1. **The theme boot step is duplicated** between `index.html` and `lib/theme.js` by
   necessity (a module cannot be imported before first paint). Both carry comments; it is
   the one place a change can silently desynchronise.
2. **With JavaScript disabled the page loads light.** The dark theme is applied by script.
3. **`--ink-quaternary` is below 4.5:1 by design** (row numbers, timestamps, the
   placeholder). If any of those ever starts carrying required information, it must move to
   a different token.
4. **Only two themes.** There is no "follow system" position in the switch itself — the
   system is followed only until the reader makes an explicit choice.

---

# Archive — the original frontend rebuild (2026-09-28)

The section below is the record of the rebuild that produced the interface the redesign
started from. It is kept for history. One decision in it — #6, "dark mode removed" — was
reversed by the redesign, on the roadmap's instruction.

## How This Was Verified

A rebuild judged on how it looks and behaves cannot be checked with `npm run build`. Two
harnesses were set up **outside `frontend/`** (in a temp directory) so no test tooling
entered the project:

1. **Headless Chrome over the DevTools Protocol**, against the production build, driving
   real interactions — clicking an example, choosing a follow-up, typing and pressing
   Enter, building real `File` objects and dispatching genuine `change` / `dragover` /
   `drop` events, reloading the page — then reading the resulting DOM back. Seven
   scenarios: conversation, verdict states, evidence, follow-ups, image input, polish,
   reload persistence — plus a four-viewport layout audit that measures overflow, column
   collapse, target sizes and scroll position numerically.
2. **Screenshots** at 1440×900, 1024×768 and 390×844, plus the About sheet, the design
   view and a full two-claim conversation.

Everything below reported as verified was observed through one of these.

## Design Decisions

The decisions that shape the whole product, recorded once:

1. **Direction — "editorial evidence desk."** Warm off-white paper (`#faf8f4`), warm
   near-black ink (`#16150f`), hairline rules as the primary structural device, almost no
   shadow, small radii. It should read like a well-set publication about evidence, not a
   console.
2. **Two typefaces, one voice.** *Instrument Serif* carries identity — hero, verdict words,
   the claim under examination. *Inter* carries the interface and reading text.
   *JetBrains Mono* is used only for small uppercase folios (section markers, counts,
   dates, source indices), which gives metadata a typeset feel.
3. **One accent.** A single cobalt (`#1b46d6`) for action, selection and focus. The only
   other saturated colours are the three verdict colours, so colour always carries meaning.
4. **Verdict palette is print-adjacent, not status-LED:** Real `#1c6b48` (moss),
   Fake `#a5341f` (oxblood), Unverified `#8a6212` (ochre), each with a soft tint and a
   hairline. No neon, no glow.
5. **Vanilla CSS with custom properties**, continuing the project's existing convention.
   Every colour, size, duration and measure lives in `tokens.css`; no component hard-codes
   a value.
6. **Light-first; dark mode removed, not inherited.** The old dark theme was a large part
   of the old product's identity, and keeping it "because it existed" is what the brief
   warns against. It was not reintroduced — **and the redesign has since reversed this**, on
   the instruction of `RAVEN_Frontend_Redesign_Phases.md` Phase 8. What was added back is a
   designed dark theme built from a token override, not the old one restored; see the
   redesign section above. The reasoning here still holds for what it was deciding: a
   theme was not carried over just because it had existed.
7. **No internal vocabulary, ever.** No intents, agent names, pipeline stages, statuses or
   model names in the interface. All user-facing language lives in `src/lib/verdict.js`.
8. **No invented product claims.** No statistics, testimonials, pricing, badges or
   marketing. The About sheet says what RAVEN does, what each verdict means, how to read
   confidence, and what the product is not.
9. **Motion only where it aids understanding** — turn entrances, a progress sweep, dialog
   fades. Nothing moves without a reason.

## Phase Log

### Phase 10 — Final Audit — Completed (2026-09-28)

Run against the production build at 1440×900 unless noted.

**Audit results — all pass:** project builds and lints clean; the main conversation works
end to end; previous messages remain visible (3 result blocks live together, an earlier
verdict still on screen after 3 further turns); Real, Fake and Unverified all render with
correct colour, confidence reading and source balance; evidence folds and unfolds per source
and all at once; follow-up actions work, including the context correctness fix; image input
works (attach, preview, remove, reject, drag, drop, image-only send, lightbox); loading and
error states work, and a failed send preserves the typed message; four viewports from 390px
to 1920px show no horizontal overflow with the composer always inside the viewport; the mock
service layer is the only path to data; **no forbidden project files were modified**; and
there are no fake statistics, testimonials, pricing or marketing elements.

**Terminology sweep — of the built bundle**, since that is what a user downloads. Zero
occurrences of `supervisor`, `evidence retrieval`, `nlp`, `pipeline`, `registry`,
`telemetry`, `system ready`, `model status`, `orchestration`, `retrieval`, `classifier`,
`embedding`, `llm`, `NEW_VERIFICATION`, `EXPLAIN_RESULT`, `MORE_EVIDENCE`, `GENERAL_QUERY`.
Remaining matches were inspected individually and are not leaks: `reflection` (×1) inside a
quoted WHO passage; `agent` (×5) from React's own `dragenter`/`dragleave`;
`follow-up` (×1) in the placeholder "Add another claim, or ask a follow-up"; `users` (×5)
from React internals and a quoted bulletin. In the source, `intents, agents, pipelines`
appear only in comments stating those words must not reach the interface.

**Marketing sweep.** Zero occurrences of `pricing`, `testimonial`, `per month`, `sign up`,
`trusted by`, `99.9`, `v1.`, `beta`, `coming soon`, `get started`. The old telemetry strip
and version badge are gone.

**Is this genuinely a new product?** The old interface and the new one share nothing but
the name and the toolchain:

| | Old | New |
| --- | --- | --- |
| Theme | Dark "obsidian console" + tactical light mode | Light-first warm paper |
| Background | Animated node-network canvas with travelling pulses | Flat paper and hairline rules |
| Type | Space Grotesk / Outfit / Jakarta + monospace telemetry | Instrument Serif display + Inter reading text |
| Chrome | Telemetry strip, `v1.4` badge, "New Docket" | Masthead rail, two quiet actions |
| Conversation | Chat bubbles | Pull-quote claim, article-style answer |
| Result | A card among cards | A 60px verdict word, no card |
| Evidence | Not presented as comparable | Three-column table with stance per source |
| Colour | Cyan glassmorphism with glow shadows | One cobalt accent, three print-adjacent verdict colours |
| Vocabulary | "CASEFILE IN PROGRESS // ACTIVE TRIANGULATION THREAD" | "Here's what I found." |

**Regression suite:** seven interaction scenarios plus the four-size layout audit re-run
against the final bundle — no page exceptions, no console errors, no failed assertions.

**Remaining limitations** (all intentional, none hidden from the user):

1. Verification is mocked. No network request is made and nothing is uploaded.
2. The conversation lives in `sessionStorage` only and disappears when the tab closes.
   There is no database and no history, and the interface never suggests otherwise.
3. `request()` has no cancellation token and no streaming; both are flagged in the README.
4. Sources, dates, excerpts and confidence figures are fixtures.
5. Animated GIFs are flattened to a still frame if they exceed the attachment limits.
6. Only the first image of a multi-file drop is used.

**Files changed:** `BUILD_PROGRESS.md` only.

### Phase 9 — Motion, Responsiveness & Product Polish — Completed (2026-09-28)

**The four-viewport measurement found real problems, not imaginary ones.**

1. **The conversation stopped following new turns** (`atBottom: false` at 1280×800). A turn
   arriving mid-smooth-scroll was read as the reader having scrolled away. Fixed by ignoring
   scroll events produced by our own scrolling for a short grace period, and making long
   jumps instant rather than animated. Re-measured: `atBottom: true` at every size.
2. **Pointer targets were too small for touch** — chips 36px, evidence toggles 21px, rail
   items 24px. Now 40px minimum below 900px (composer controls 42px). Desktop keeps its
   tighter rhythm, which is correct for a mouse.
3. **Evidence passages shrank to 13px on phones** — the text a verdict is actually checked
   against was the smallest thing on screen. Now stays at 15px.
4. **Sending while RAVEN was working discarded the message.** `useConversation` ignored a
   submission while busy, but the composer cleared its field regardless — silent data loss.
   The send control now disables itself with an explanatory `title` and a stray Enter
   preserves what was typed. Verified: text preserved, no duplicate turn, still present when
   the check finished.
5. **The mobile media query overrode the long-claim size** (found in Phase 4, fixed here
   against the real viewport).

**Polish added:** a **"Latest answer"** control that floats above the composer when the
reader has scrolled back through a long conversation (verified: appears on scroll-up,
returns to the newest turn, then disappears); **session persistence** in `sessionStorage`
capped at 1.5M characters, so an accidental reload does not lose a check that is still open
— deliberately not `localStorage`, and never implying history (verified across a real page
reload: 2 turns, verdict, 3 source rows and working context-correct follow-ups restored); a
typographic favicon matching the wordmark; removal of the unused default Vite assets;
keyboard parity for the rail underline.

**Motion stays restrained:** turn entrances (7px rise, 620ms), the progress sweep, dialog
fades. `prefers-reduced-motion` disables all of it through the global override in
`reset.css` — confirmed present in the shipped CSS bundle.

**Known issues.** Above 900px the smallest controls are 21–36px, which would be wrong on a
touch device wider than 900px (a tablet in landscape); a pointer-coarse query would be
better than a width query. `#design` and `#about` are reachable by hash as development aids
only — the product has no navigation to them.

**Files changed:** `src/hooks/useScrollAnchor.js` (rewritten), `src/pages/ChatView.jsx`,
`src/components/Composer.jsx`, `src/lib/session.js` (new), `src/hooks/useConversation.js`,
`src/styles/{shell,primitives,evidence,composer,empty-state,conversation}.css`,
`index.html`, `public/favicon.svg` (new); removed `public/vite.svg` and
`public/test-image.png`.

### Phase 8 — Mock Service Layer & Backend-Ready Boundary — Completed (2026-09-28)

**Built.** One seam instead of four functions: the interface makes every request through
`request({ kind, payload })` with four kinds — `assessment`, `explanation`, `evidence`,
`answer`. Previously the hook imported four named service functions and mapped actions to
them, which leaked the mapping out of the boundary. The whole simulation is now one visible
block (`MOCK = { enabled, latency, failureTrigger }`), with a marked insertion point where
the real call goes and a commented `fetch` sketch that assumes no method, path or streaming
model — **no backend endpoint is invented or modified anywhere in the frontend**.
`README.md` was rewritten (it still documented the *old* frontend) and now carries the full
shape reference: every request kind with its payload, all five response types, the error
turn, and the four steps to swap in a real backend. The unused `react-router-dom` dependency
was removed.

**Loading and failure behaviour.** Per-interaction latency ranges (1.1–2.6s) rather than one
flat delay; submissions locked to one at a time. `RavenError` carries a `userMessage` written
for a person, a machine `code` and a `retryable` flag; unknown request kinds raise a
non-retryable error. A reachable failure path (`simulate error`) is documented so the error
and retry states can be demonstrated on demand.

**Checks.** Build and lint clean after the refactor; the full Phase 3 conversation scenario
re-run against the rebuilt bundle behaved identically through the new seam;
`grep -rln "mockData" src --include=*.jsx` returns nothing, so no component can see mock data.

**Known issue.** `request()` has no cancellation token — unnecessary while submissions are
serialised, but a streaming backend will want one, and the README says so.

**Files changed:** `src/services/raven.js` (rewritten), `src/hooks/useConversation.js`,
`README.md`, `package.json` / `package-lock.json`.

### Phase 7 — Multimodal Input — Completed (2026-09-28)

**Built.** `src/lib/image.js` reads a `File` into `{ dataUrl, name, width, height }`, rejects
non-images and files over 8 MB with messages written for a person, and **downscales anything
larger than 1600px or heavier than 1.5 MB** — a 12-megapixel phone photo becomes a ~7 MB
string as a data URL and would otherwise be carried through every render. `Composer` supports
multiline typing, an attach button, a preview strip with file name and pixel dimensions, a
remove control, `role="alert"` rejection messages, a drop overlay, and **paste**. Drag and
drop is bound to the **window**, not the 200px field, with `preventDefault`, so dragging a
file onto the page attaches it instead of the browser navigating away. Image-only,
text-only and image-plus-text all send; the placeholder and the send button's label change
when an image is attached. `ImageLightbox` opens the attachment at full size from the
conversation entry.

**Decisions.** No toolbar — one attach button, one field, one send; the preview lives inside
the composer as a single strip. Attachments are never uploaded: they are read locally and
passed to the service layer as a data URL that the mock uses only to choose a scenario. The
drop overlay covers the composer, but the drop *target* is the whole window.

**Checks (CDP, building real PNG `File` objects and dispatching genuine events).** Attach →
preview with `screenshot.png`, `240 × 240` and a `data:image/png;base64,…` thumbnail, send
enabled. Remove → preview gone, send disabled. Attach `notes.txt` → rejected with "That file
isn't an image…" and `role="alert"`, no preview created. `dragover` → overlay appeared;
`drop` → overlay cleared and `dropped-photo.png` attached. **Send with no text at all** →
composer cleared, user turn contains the image and no text, thinking turn appeared; the
answer came back `Fake` with the interpretation labelled **"Read from the image"**, which is
the case that label exists for. Lightbox opened the right image and Escape closed it.

**Known issues.** Animated GIFs are flattened to a still if they exceed the size limits.
Only the first image of a multi-file drop is used, and the interface does not yet say so.

**Files changed:** added `src/lib/image.js`, `src/components/ImageLightbox.jsx`,
`src/styles/lightbox.css`; rewritten `src/components/Composer.jsx` and
`src/styles/composer.css`; changed `src/components/conversation/UserTurn.jsx`,
`src/styles/{conversation,main}.css`, `src/pages/ChatView.jsx`.

### Phase 6 — Natural Follow-Up Interaction — Completed (2026-09-28)

Most of this interaction was built with the conversation itself (Phase 3) and the source
questions (Phase 5), so this phase was about making it *correct* and quiet rather than adding
more buttons.

- **Fixed a real bug:** suggestions were dispatched using the most recently checked claim, so
  with two claims in one conversation, clicking the first answer's "Why this verdict?"
  explained the *second* claim. Each suggestion now carries the claim it belongs to, stamped
  when its turn is created, falling back to the latest context only if it has none.
- Suggestions live inside the answer they belong to, in prose order — not in a global action
  bar. Only the newest answer offers them at full strength; earlier ones sit at 72% opacity
  and return to full on hover or keyboard focus.
- **"Check another claim" is not a turn.** It focuses the composer and shows a quiet line
  rather than putting words in the user's mouth. This is a deliberate deviation from a
  literal reading of the phase; the requirement — that the earlier result is not destroyed
  and the conversation continues naturally — is met. "What is this source?" and "Show me more
  evidence" *do* become turns, because they have something to ask.
- No follow-up is ever labelled with an internal intent.

**Checks (CDP, 1440×900).** Two claims → 2 result blocks, 1 loud suggestion group, 1 quiet.
Clicked the **older** "Why this verdict?" → the answer used the older claim (asserted against
the expected reasoning text), confirming the bug is fixed. Three claims → 3 result blocks with
the newest still loud. "Check another claim" → **turn count unchanged (8 → 8)**, nudge line
appeared, composer focused, placeholder reflected the new state.

**Known issue.** After a long conversation, older suggestion groups remain clickable. They are
now context-correct, so this is a density question rather than a correctness one.

**Files changed:** `src/hooks/useConversation.js`, `src/components/conversation/FollowUps.jsx`,
`src/components/conversation/RavenTurn.jsx`.

### Phase 5 — Evidence Presentation — Completed (2026-09-28)

**Built.** `EvidenceSection` presents sources as a small three-column table rather than a wall
of cards: index | publisher, title, passage, actions | what it means for the claim (stance,
date, domain). Passages fold to three lines with `-webkit-line-clamp` and expand per source,
with an **"Open all passages" / "Fold all passages"** control for comparing side by side;
`aria-expanded` is set on every toggle. Every source shows **stance — supports / contradicts /
doesn't settle it** — colour-coded with a hollow dot for "unclear". That is what makes a
verdict explainable: the interface can show that two sources contradict a claim rather than
merely listing sources. Titles link to the original in a new tab; the domain is shown without
`https://www.` noise; dates are relative ages with the full date in `title`. A wider search
that re-surfaces a source already listed marks it **"Already shown"** instead of appearing to
discover it twice, computed by walking the conversation's own turns. "Further sources" is used
as the heading when the turn came from "Show me more evidence".

**Decisions.** Readable first: passage text is 15px/1.62 in secondary ink inside a real
`<blockquote>` with a hairline quote rule. Folded by default — five sources × 90 words is
~1,900 words on screen; folded, the same list is scannable in seconds. The claim's stake in
each source is right-aligned in its own column so the eye can run down it. On small screens
that column moves *above* the source, so what a source says about the claim is read first.

**Checks (CDP, 1440×900).** `Sources — 03` with 3 rows; first row exposed index `01`,
publisher `World Health Organization`, kind `Primary statement`, stance `Supports the claim`,
domain `who.int`, link `https://www.who.int/…`, `target="_blank"`. Folded: `-webkit-line-clamp: 3`,
73px tall, from 590 characters. Expanded: `line-clamp: none`, 219px, label "Fold the passage",
`aria-expanded` `false → true`. Open all → 3; fold all → 0. "What is this source?" became a
normal turn and the earlier result block and its 3 rows were still on the page. "Show me more
evidence" → `Further sources — 02` with one row marked "Already shown".

**Known issues.** Passages longer than ~800 words were not tested (the fold handles them the
same way). Relative ages are computed at render and do not tick during a long session.

**Files changed:** added `src/components/result/EvidenceSection.jsx`, `src/styles/evidence.css`;
changed `src/components/conversation/{RavenTurn,Conversation}.jsx`,
`src/services/mockData.js` (new "further sources" fixtures for every scenario),
`src/services/raven.js`, `src/styles/main.css`.

### Phase 4 — Verification Result Experience — Completed (2026-09-28)

**Built.** `AssessmentCard` is the verdict block: a hairline, the folio "The verdict", then the
verdict word at up to **60px in Instrument Serif** in the verdict colour with a status dot, the
stance sentence beside it, a hairline, then a two-column reading — **Confidence** (word +
percentage + 2px meter + a sentence) and **How the sources line up** (a stacked hairline bar
with "3 support" / "1 contradict · 2 don't settle it"). **"As I understood it"** shows the claim
as RAVEN read it, but only when that differs from what the user actually wrote, or labelled
**"Read from the image"** when the claim came from an attachment — otherwise the block would
repeat the user's own turn back at them. The Phase 3 placeholder verdict line and its CSS were
deleted.

**Decisions.** No card, no tint, no glow: colour and size carry the authority, with whitespace
doing the rest. A tinted result panel would have been the dashboard look the brief warns
against. Verdict colour appears at exactly two sizes — the verdict word and a 6px dot — so it
reads as meaning, not decoration. Confidence is never a bare percentage: it is a word
(`High`/`Moderate`/`Low`/`Very low`) and a sentence, with the number the smallest element in the
block. The sentence is **verdict-aware** — high confidence in a Fake verdict says the sources
are consistent about why the claim is wrong, which is a different statement from sources
agreeing it is true. That distinction was caught in review and fixed.

**Checks (CDP, 1440×900).** Real: `rgb(28, 107, 72)`, word 60px, `High`, `93%`, meter 93%.
Fake: `rgb(165, 52, 31)`, `High`, `97%`, corrected confidence sentence. Unverified:
`rgb(138, 98, 18)`, `Low`, `42%`, balance `1 contradict · 2 don't settle it` in 2 segments with
the hollow status dot. All three result blocks stayed on the page together (`1 → 2 → 3`). An
88-character forwarded message took the `turn__claim--long` class — which exposed a bug (see
Phase 9, item 5) where the mobile media query overrode the long-claim size.

**Files changed:** added `src/components/result/AssessmentCard.jsx`,
`src/styles/assessment.css`; changed `src/components/conversation/Conversation.jsx`,
`src/lib/verdict.js`, `src/styles/{conversation,main}.css`.

### Phase 3 — Conversational Experience — Completed (2026-09-28)

**Built.** `src/services/raven.js` — **the service boundary, introduced here rather than in
Phase 8 on purpose.** A conversation cannot be built or tested without responses, and the
alternative was throwaway inline mocks in components that Phase 8 would then have to unpick.
Phase 8 hardened and documented it rather than introducing it. Alongside it: `mockData.js`
(twelve evidence items with 60–90 word excerpts, plus publisher, URL, date, kind and stance;
five scenarios); `lib/verdict.js` (all user-facing language in one place); `lib/format.js`;
`hooks/useConversation.js` (turns, in-flight state, claim context, dispatch — failures become
turns, so an error is part of the record and retryable); `hooks/useScrollAnchor.js`; and the
components `Conversation`, `UserTurn`, `RavenTurn`, `ThinkingTurn`, `ErrorTurn`, `FollowUps`,
`EmptyState`, `Composer`, `ChatView`.

**Decisions.** The user's claim is set as **display type, not a bubble** — a hairline down the
left, serif at up to 1.8rem, dropping to reading size for anything over 110 characters so a
forwarded paragraph does not become a wall. Answers are prose beneath a speaker line made of a
small serif `Raven` and a hairline rule; that single device is what stops the page reading as a
generic chat log. **No streamed typing animation** — RAVEN answers in complete paragraphs;
typing effects would be theatre. Progress is language, not a spinner: four plain-language
stages that settle on the last one, over a 1px sweeping rule, with no invented percentage.
Older suggestions recede.

**Checks (CDP, production build).** Empty state showed 3 examples and the composer; picking one
produced the user turn plus a thinking turn reading "Looking for sources"; 4.2s later the answer
carried verdict `Real`, the stance line, the lead, **3 reasoning paragraphs** and exactly the
three expected follow-ups. "Why this verdict?" produced a 4-turn conversation with **2 user and
2 RAVEN turns, the earlier verdict still on screen**, and the earlier follow-ups marked quiet —
the core requirement of this phase. Typing `simulate error` and pressing Enter sent, cleared the
composer, and produced a failure turn 4.2s later with "Not completed", the honest message and a
working "Try again". No console output, no page exceptions.

**Note on a false alarm:** `ThinkingTurn` deliberately shares `.turn--raven` for visual
continuity, so raw `.turn--raven` counts include the placeholder. Verified against the real turn
list rather than the count.

**Files changed:** added `src/services/{raven,mockData}.js`, `src/lib/{verdict,format}.js`,
`src/hooks/{useConversation,useScrollAnchor}.js`, `src/components/{Composer,EmptyState}.jsx`,
`src/components/conversation/*`, `src/pages/ChatView.jsx`,
`src/styles/{conversation,composer,empty-state}.css`; changed `src/App.jsx`,
`src/components/AppShell.jsx`, `src/styles/main.css`.

### Phase 2 — New Application Shell — Completed (2026-09-28)

**Built.** `AppShell` — a masthead rail and one continuous workspace, with `children` filling the
scrolling region and `dock` holding the composer. `Wordmark` — the identity mark as pure
typography: `Raven` in Instrument Serif, uppercase, widely tracked. No bird, no glyph, no
gradient, since the brief rules out decorative crow/AI imagery. `AboutSheet` — the product's only
secondary surface, explaining what RAVEN does, what each verdict means, what an answer contains,
how to read confidence and what RAVEN is **not**, with full dialog behaviour (scrim, Escape,
focus moved in and restored, Tab trap). `IntroStatement` — the opening proposition, reused as the
top of the empty state rather than rebuilt. Plus `shell.css` and `intro.css`.

**Decisions.** **Rail over header:** a left masthead rail gives identity real presence and reads
like the spine of a publication, where a top header would look like every other SaaS app.
**One scrolling region, one docked composer** (a flex item, not a floating overlay) is more
predictable on mobile with the on-screen keyboard and means nothing hides behind the composer.
**Measure 46rem** — wide enough for a serif display statement, narrow enough to read reasoning
and evidence comfortably. **Nav is two items, sometimes one:** `About` always, `Start a new
check` only when there is a conversation to leave. No History, because there is no persistence
and the brief rules out implying one. The telemetry strip, version badge and "New Docket"
language were removed; the rail footer instead states something true and useful.

**Checks.** Build succeeded; lint clean (a `no-unused-vars` error from a polymorphic `as` prop on
Wordmark was fixed rather than suppressed). Headless Chrome DOM dumps at `/` and `/#about`
confirmed `.shell`, `.rail`, `.wordmark--md`, `.workspace__column--centered` and `.intro__title`
all mount, with `.sheet__panel` present only on `/#about` — so the dialog mounts and opens without
throwing. Screenshots at 1440×900 and 390×844 checked the rail collapsing to a masthead bar and
the dialog becoming a bottom sheet.

**Known issues.** Touch-device rail hover states reviewed in Phase 9. The workspace content was
still a static statement until Phase 3.

**Files changed:** added `src/components/{AppShell,Wordmark,AboutSheet,IntroStatement}.jsx`,
`src/styles/{shell,intro}.css`; changed `src/App.jsx`, `src/styles/main.css`.

### Phase 1 — New Design System — Completed (2026-09-28)

**Built.** `tokens.css` — the whole design language as custom properties: typefaces, paper/ink
scale, hairline rules, accent, three verdict palettes, a fluid display type scale
(`--text-hero` → `--text-micro`), leading and tracking, a 4px-based space scale with editorial
jumps at the top end, radii, near-invisible shadows, motion easings and durations, layout
measures and z-index layers. `base.css` — document defaults, an ink focus ring that is never
removed, thin ink scrollbars. `primitives.css` — `.folio`, `.rule`, `.surface`, `.btn` (with
primary, accent, quiet, bare, icon, three sizes, `--block`, disabled, active and a width-stable
loading state), `.field` with hover/accent-focus/`aria-invalid` styling, `.chip`, `.tag`,
`.meter` and `.dot`. Plus `DesignSystem.jsx` and `design-system.css` — an internal review view at
`#design` showing the type scale, swatches, verdict palette, every control state, chips, tags,
meters and the spacing ruler.

**Decisions.** Display type is deliberately oversized and tightly tracked
(`clamp(2.5rem, 6.2vw, 4.75rem)`, `-0.022em`), because the brief asks the page to feel strong
with very little on it — the empty state leans on type rather than graphics. Reading text is 17px
at 1.62 leading: body copy is the product here, so measure and leading were set for reading
comfort. Radii stay small (3–14px); nothing is glassy. Depth comes from exactly three devices —
hairline rules, a subtle paper/tint shift, and a barely-visible shadow. The three verdict colours
were chosen and reviewed as a set so none of them reads as an error state.

**Checks.** Build succeeded (CSS 13.07 kB / gzip 3.56 kB); lint clean. Served the production build
and loaded `#design` in headless Chrome: all seven sections rendered with no empty root, so
nothing threw at runtime. Screenshot captured for manual review.

**Known issues.** Visual verification at this point was DOM- and build-level; pixel-level review of
the design view is worth a human pass. The `#design` route is a development aid and is excluded
from product navigation.

**Files changed:** added `src/styles/{tokens,base,primitives,design-system}.css`,
`src/pages/DesignSystem.jsx`; changed `src/styles/main.css`, `src/App.jsx`, `src/main.jsx`,
`index.html`.

### Phase 0 — Frontend Reset & Progress System — Completed (2026-09-28)

**What the old frontend was.** A dark "tactical console": an animated full-screen node-network
`<canvas>` background with travelling signal pulses; a header telemetry strip reading
`EVIDENCE MATRIX: 42 REGISTRIES SYNCHRONIZED`; conversation chrome labelled
`CASEFILE IN PROGRESS // ACTIVE TRIANGULATION THREAD`; a `v1.4` badge and a "New Docket" button;
and a glassmorphic neon token set. Every one of those is explicitly listed under **Avoid** in
section 2 of the brief (glowing AI aesthetics, huge animated network backgrounds, fake product
statistics, excessive glassmorphism, technical jargon in the UI). There was no design system worth
preserving, so this was a genuine reset rather than a re-skin.

**What was done.** Inspected the existing frontend *before* deleting anything so the decision is
recorded rather than assumed, then removed the old UI implementation: all ten components, both
pages, all nine old stylesheets (≈2,200 lines of CSS), the old mock service, the old entry files,
`src/assets/react.svg`, and `dist/` so no stale bundle could be served. Kept the configuration
that still serves the new frontend: `package.json`, `vite.config.js`, `eslint.config.js`,
`.gitignore`, `package-lock.json`. Rebuilt a clean entry foundation — `index.html` with a new
title, description and editorial font stack; `src/main.jsx`; a placeholder `src/App.jsx`; and
`src/styles/reset.css` — and created this progress file.

**Decisions.** `react-router-dom` dropped from use: the brief asks for one continuous
conversational surface, not disconnected pages. Light-first, with a warm paper `theme-color`
rather than a dark console. Dark mode removed rather than carried over. Fonts loaded once in
`index.html` and referenced only through tokens, so swapping the stack touches one file.

**Checks.** `npm run build` succeeded (29 modules) and `npm run lint` was clean; `git status`
confirmed only paths under `frontend/` were changed.

**Files changed:** deleted `src/components/*`, `src/pages/*`, `src/styles/*`,
`src/services/ravenApi.js`, the old `src/App.jsx` and `src/main.jsx`, `src/assets/react.svg`,
`dist/*`; added `BUILD_PROGRESS.md`, `src/main.jsx`, `src/App.jsx`, `src/styles/reset.css`;
rewrote `index.html`.

## Next Work (not phases of this rebuild)

1. **Connect a real backend** — replace the body of `request()` in `src/services/raven.js` and
   delete `mockData.js`. The contract is documented in `frontend/README.md`.
2. **Streaming answers**, if the backend supports them. Only `useConversation` needs to learn
   about partial turns; the components already render whatever a turn contains.
3. **A pointer-coarse media query** so larger tap targets apply to touch devices at any width.
4. **Real image understanding** — attachments are read, downscaled and displayed, but the mock
   uses only their presence to choose a scenario.
5. **Persistent history**, only if a store actually exists to back it.
