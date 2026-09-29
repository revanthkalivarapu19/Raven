# RAVEN — frontend

A conversational evidence-verification interface. Someone brings a claim — typed,
pasted from a forward, or attached as a screenshot — and RAVEN looks for sources
that speak to it, then shows which way they point, why, and what they actually say.

The interface is built around **one continuous conversational surface**. A new claim
is a new turn in the same conversation: nothing is replaced, and an earlier result
stays readable while a later one is being checked.

---

## Running it

```bash
cd frontend
npm install
npm run dev      # development server
npm run build    # production build into dist/
npm run preview  # serve the production build
npm run lint
```

## Stack

- **React 19 + Vite** — no meta-framework, no router. The product is a single
  surface, so routing would exist only to hide things.
- **Vanilla CSS with custom properties.** All design values live in
  `src/styles/tokens.css`; components never hard-code a colour, size or duration.
  No utility framework and no CSS-in-JS, which keeps the design system in one
  readable place and the bundle small.
- **`lucide-react`** for the handful of interface icons that are needed.
- No state library. Conversation state is React state in one hook; the theme is
  one hook and a `data-theme` attribute on `<html>`.
- **No dependencies were added for the redesign.** The confidence ring is inline
  SVG and the theme switch is CSS, so neither needed a charting or icon package.

## Layout

```
src/
  App.jsx                 route switch (product, and two internal review views)
  pages/
    ChatView.jsx          the product: conversation state + shell wiring
    DesignSystem.jsx      internal review view of the design primitives (#design)
  components/
    AppShell.jsx          rail + workspace frame, About sheet
    Wordmark.jsx  AboutSheet.jsx  IntroStatement.jsx
    ThemeToggle.jsx       the light/dark switch
    EmptyState.jsx        the opening proposition and example claims
    Composer.jsx          text + image input
    ImageLightbox.jsx
    conversation/         Conversation, UserTurn, RavenTurn, ThinkingTurn,
                          ErrorTurn, FollowUps
    result/               AssessmentCard (the verdict), ConfidenceRing,
                          EvidenceSection (sources)
  hooks/
    useConversation.js    turns, in-flight state, claim context, dispatch
    useScrollAnchor.js    follow new turns, unless the reader scrolled up
    useTheme.js           current theme + the switch for it
  lib/
    verdict.js            all user-facing language: verdicts, confidence, stances
    format.js             dates, ages, percentages, domains, text trimming
    image.js              reading, validating and downscaling attachments
    theme.js              reading, storing and applying the theme preference
  services/
    raven.js              THE BOUNDARY — one request function
    mockData.js           mock fixtures (deleted when a backend exists)
  styles/
    tokens.css  base.css  primitives.css  shell.css  intro.css
    empty-state.css  conversation.css  composer.css  assessment.css
    evidence.css  lightbox.css  design-system.css
```

Two internal views exist for review and are not part of the product:
`#design` shows the type scale, both themes, the confidence ring at every reading
and every control state; `#about` opens the About sheet directly.

---

## The service boundary

Every turn in the conversation comes from **one function**:

```js
import { request } from './services/raven'

const turn = await request({ kind, payload })
```

`src/services/raven.js` is the only file that knows how an answer is produced.
Components never decide a verdict, build evidence or write reasoning.

### Request kinds

| `kind`        | When it is used                       | `payload`                                        |
| ------------- | ------------------------------------- | ------------------------------------------------ |
| `assessment`  | A new claim is submitted              | `{ text: string, image: Attachment \| null }`     |
| `explanation` | "Why this verdict?"                   | `{ claim: string, verdict: Verdict }`             |
| `evidence`    | "Show me more evidence"               | `{ claim: string }`                               |
| `answer`      | "What is this source?"                | `{ source: EvidenceItem }`                        |

### Response shapes

A response is a **turn** — the same object the conversation stores and renders.

```ts
type Verdict = 'Real' | 'Fake' | 'Unverified'
type Stance  = 'supports' | 'contradicts' | 'unclear'

type AssessmentTurn = {
  kind: 'assessment'
  verdict: Verdict
  intro: string              // one human lead-in sentence
  claim: string              // the claim as RAVEN understood it
  confidence: number         // 0–1
  reasoning: string[]        // paragraphs, shown as prose
  evidence: EvidenceItem[]
  followUps: FollowUp[]
}

type ExplanationTurn = {
  kind: 'explanation'
  verdict: Verdict
  intro: null
  reasoning: string[]
  followUps: FollowUp[]
}

type EvidenceTurn = {
  kind: 'evidence'
  verdict: Verdict
  intro: string
  evidence: EvidenceItem[]
  note: string | null        // what the wider search did *not* find
  followUps: FollowUp[]
}

type AnswerTurn = {
  kind: 'answer'
  intro: string
  reasoning: string[]
  followUps: FollowUp[]
}

type EvidenceItem = {
  id: string                 // stable across turns; used to mark repeats
  publisher: string          // who published it
  title: string
  url: string                // opened in a new tab
  published: string          // ISO date
  kind: string               // "Primary statement", "Wire report", …
  stance: Stance             // what it says about the claim
  excerpt: string            // the passage actually used, can be long
}

type FollowUp = {
  id: string
  label: string              // human-facing, e.g. "Why this verdict?"
  action: 'why' | 'evidence' | 'another'
}

type Attachment = {
  dataUrl: string            // stays local; never uploaded by the mock
  name: string
  width: number
  height: number
}

type Turn =
  | AssessmentTurn | ExplanationTurn | EvidenceTurn | AnswerTurn
  | { kind: 'error', message: string, retryable: boolean,
      failed: { kind: string, payload: object } }
```

Failures are the one case that does not come from a response: `request()` throws
`RavenError`, which carries a `userMessage` written for a person, a `code`, and a
`retryable` flag. The conversation converts that into an `error` turn so a failure
is part of the record and can be retried in place rather than replacing anything.

### Opening suggestions

`getStarters() → { id, label, detail }[]` — the examples offered on the empty screen.

### Replacing the mock with the real backend

1. Set `MOCK.enabled = false` in `src/services/raven.js` (or delete the mock
   branches), and put the real call inside `request()` where the comment marks it.
2. Keep `request()`'s signature and returned shapes. Every component keeps working.
3. Delete `src/services/mockData.js` — nothing else imports it.
4. If the backend streams, the shape of the *turn* can still be produced
   incrementally; only `useConversation` needs to learn about partial turns.

Nothing outside `frontend/` knows about any of this, and no backend endpoint is
assumed by the frontend.

### Testing hooks

- A message containing **`simulate error`** makes the next `assessment` fail, so the
  error and retry states can be reached on purpose.
- Scenario selection is keyword-based (`detectScenario` in `mockData.js`):
  gold/RBI → Fake, strike/transport/Bengaluru → Unverified, WHO/mpox/emergency →
  Real, an image on its own → the image claim, anything else → Unverified.

---

## Design system

`#design` (`src/pages/DesignSystem.jsx`) renders the type scale, paper and ink
swatches, the verdict palette, every button and field state, chips, tags, source
stances, the confidence ring at five readings, and the spacing ruler — the fastest
way to check that a change belongs to the product.

The direction is **editorial evidence desk**: warm paper, near-black ink, hairline
rules as the main structural device, almost no shadow, small radii, a single cobalt
accent (`--accent`), and three print-adjacent verdict colours that are the only
saturated colours in the interface — so colour always means something.

### How a result is composed

The verdict block is one editorial statement rather than a stack of cards, and it
reads in the order a person needs it:

1. **The verdict** — one word at display size, in the verdict colour.
2. **The stance** — the sentence that word is short for, set in the display serif.
3. **RAVEN's opening line** — also display serif, sitting with the verdict it
   qualifies rather than floating above the reasoning.
4. **Confidence** — a ring whose arc length *is* the reading, coloured by the
   verdict so the figure never reads as a free-floating score. The ring draws
   itself on arrival.
5. **How the sources line up** — a hairline legend, not a chart.
6. **The claim as understood** — set in italic display type, shown only when
   RAVEN's reading differs from what was typed or came from an image.
7. **The explanation** — labelled ("Why I read it this way", "The reasoning",
   "What the further sources show", "About this source") so a section of prose is
   never an undifferentiated block. Labels live in `lib/verdict.js`.
8. **The sources** — a small table, one row per source, passage folded to three
   lines until asked for.

## Themes

Light and dark are two sets of **semantic colour tokens**, not two stylesheets:

- `:root` in `tokens.css` holds the light set; `[data-theme='dark']` overrides the
  colours only.
- Type, space, radii, motion and layout are declared once and are theme-independent.
- Colour is read through tokens everywhere. If you find a literal colour outside
  `tokens.css`, that is a bug.
- The dark set is designed rather than inverted: a warm near-black paper, an accent
  *lifted* so it stays legible on dark (a cobalt dark enough for white paper is
  unreadable on black), verdicts re-mixed at lower luminance, and depth carried by
  hairlines because shadows stop reading on dark grounds.

Preference: the first visit follows the operating system; using the switch stores an
explicit choice in `localStorage` (`raven:theme:v1`), which then wins, including over
later system changes. `localStorage` is right for a standing preference — the
conversation deliberately uses `sessionStorage` instead, because a piece of work
should not outlive the tab and a preference should.

`index.html` carries a small inline copy of the boot step so the correct theme is
set before the first paint and a dark-theme reader never sees a white flash. **That
copy duplicates the key and the attribute from `src/lib/theme.js` and must be kept
in step with it.**

### Contrast

The two quiet ink steps are held to a floor rather than picked by eye:
`--ink-tertiary` (notes, labels, folios, source dates and types) clears 4.5:1 on the
paper in both themes, and `--ink-quaternary` is reserved for marks that are genuinely
supplemental — row numbers, timestamps, the composer placeholder.

## Accessibility

- Dialog focus is trapped and restored; Escape closes (`AboutSheet`, `ImageLightbox`).
- Every toggle reports `aria-expanded`; every icon-only control has a label. The theme
  switch is a `role="switch"` with `aria-checked`.
- Text and interface contrast is verified in both themes; see Contrast above.
- The conversation follows new turns without moving the page under a reader who has
  scrolled up (`useScrollAnchor`).
- `prefers-reduced-motion` disables all animation globally (`reset.css`), including
  the confidence ring's draw.

## Known limitations

- Verification is mocked (see above). Nothing is uploaded, and no network call is made.
- The conversation is held in `sessionStorage`: an accidental reload does not lose an
  open check, and closing the tab does. There is no database and no history, and the
  interface does not imply otherwise. Conversations over ~1.5 MB (large attachments)
  are simply not persisted rather than truncated.
- Confidence figures, sources and dates are fixtures, not results.
- With JavaScript disabled the interface loads in the light theme; the dark theme is
  applied by script.

## Two deliberate rules

Two rules hold the interface together:

1. **No internal vocabulary reaches the user.** No intents, agent names, pipeline
   stages, statuses or model names exist in the interface. All user-facing language
   lives in `src/lib/verdict.js`.
2. **No invented product claims.** No statistics, testimonials, pricing, badges or
   marketing sections. The About sheet describes what RAVEN does, what each verdict
   means, how to read confidence, and what the product is not.
