/**
 * Mock content for the RAVEN service layer.
 *
 * This is realistic fixture data, not product copy: claim wording, reasoning
 * prose, and source excerpts are written long enough to test the interface
 * against real-length content. Nothing here reaches the UI directly — only
 * through `raven.js`.
 *
 * When the real backend is connected, this file is the only thing that goes
 * away. See the request/response shapes documented in README.md.
 */

/* -------------------------------------------------------------------------
   Evidence
   `stance` is what the source says about the claim: supports / contradicts /
   unclear. It is what lets the interface show *why* a verdict landed where it
   did instead of just counting sources.
   ------------------------------------------------------------------------- */
const E = {
  whoStatement: {
    id: 'src-who-mpox',
    publisher: 'World Health Organization',
    title: 'Mpox no longer constitutes a public health emergency of international concern',
    url: 'https://www.who.int/news/item/mpox-pheic-status',
    published: '2023-05-11',
    kind: 'Primary statement',
    stance: 'supports',
    excerpt:
      'The Director-General concurs with the advice of the Emergency Committee that the multi-country outbreak of mpox no longer constitutes a public health emergency of international concern. In reaching this conclusion, the Director-General considered the sustained decline in reported global cases, the absence of a change in severity or clinical presentation, and the improvements in access to diagnostics and vaccines across affected regions. The Director-General noted that the designation is a reflection of current transmission patterns rather than a statement that the outbreak has ended.',
  },
  reutersMpox: {
    id: 'src-reuters-mpox',
    publisher: 'Reuters',
    title: 'WHO declares end of mpox emergency after case numbers fall sharply',
    url: 'https://www.reuters.com/business/healthcare-pharmaceuticals/who-declares-end-mpox-emergency',
    published: '2023-05-11',
    kind: 'Wire report',
    stance: 'supports',
    excerpt:
      'LONDON — The World Health Organization said on Thursday it was ending the global health emergency it declared for mpox, saying the outbreak had come under control more than a year after it began spreading rapidly. Reported cases had fallen by about ninety per cent from their peak, the agency said. The decision follows a meeting of the emergency committee convened a day earlier, which advised the Director-General that the criteria for maintaining the designation were no longer met.',
  },
  apMpox: {
    id: 'src-ap-mpox',
    publisher: 'Associated Press',
    title: 'WHO downgrades mpox outbreak, keeping watch on transmission',
    url: 'https://apnews.com/article/who-mpox-emergency-ends',
    published: '2023-05-12',
    kind: 'Wire report',
    stance: 'supports',
    excerpt:
      'GENEVA — Health officials cautioned that ending the emergency designation does not mean the virus has been eliminated, and that surveillance will continue in the countries most affected. Several national health agencies said they would maintain monitoring and vaccination programmes for at-risk groups. Officials described the change as a shift in classification rather than the end of the response.',
  },
  pibFactCheck: {
    id: 'src-pib-rbi-gold',
    publisher: 'Press Information Bureau — Fact Check Unit',
    title: 'Claims of RBI giving gold coins to savings account holders are false',
    url: 'https://pib.gov.in/FactCheck/rbi-gold-coins-claim',
    published: '2024-02-19',
    kind: 'Government fact check',
    stance: 'contradicts',
    excerpt:
      'A message circulating on messaging applications claims that the Reserve Bank of India is distributing free gold coins to savings account holders as an anniversary benefit, and instructs recipients to click a link to claim them. The government fact check unit found no such scheme. The link in the message leads to a site that imitates the appearance of an official banking portal and asks for account credentials and a one-time password. The public is advised not to enter any banking details on such pages.',
  },
  rbiPress: {
    id: 'src-rbi-press',
    publisher: 'Reserve Bank of India',
    title: 'RBI cautions public against fraudulent offers in its name',
    url: 'https://www.rbi.org.in/pressrelease/fraudulent-offers',
    published: '2024-01-30',
    kind: 'Primary statement',
    stance: 'contradicts',
    excerpt:
      'The Reserve Bank of India has issued repeated cautions that it does not open accounts for individuals, does not give out prizes, reward money, or benefits of any kind, and does not ask members of the public to share account details, passwords, or one-time passwords. Members of the public are advised to verify the credentials of any person or entity claiming to represent the Reserve Bank before sharing information or making payments.',
  },
  factChecker: {
    id: 'src-factcheck-india',
    publisher: 'FactCheck India',
    title: 'No, the RBI is not crediting gold to savings accounts',
    url: 'https://www.factcheckindia.org/rbi-gold-coins',
    published: '2024-02-21',
    kind: 'Independent fact check',
    stance: 'contradicts',
    excerpt:
      'Contacted for comment, a spokesperson for the national banking association said no such scheme exists and that no representation was made to the association by the central bank. The domain in the forwarded link was registered eleven days before the message began circulating and has no connection to any licensed bank. Several recipients reported that the page asked for an account number and an OTP before a countdown timer expired.',
  },
  strikeNews: {
    id: 'src-strike-news',
    publisher: 'The Hindu',
    title: 'Transport associations say strike talks continue, no final decision',
    url: 'https://www.thehindu.com/news/cities/bangalore/transport-strike-talks',
    published: '2026-09-24',
    kind: 'Local report',
    stance: 'unclear',
    excerpt:
      'Talks between the transport federation and the regional transport authority remained inconclusive on Wednesday, with both sides describing the discussions as constructive. A federation office-bearer said an announcement would follow a general body meeting, declining to confirm whether services would be withdrawn. The authority said it had asked operators to keep services running while discussions continue.',
  },
  strikePrint: {
    id: 'src-strike-print',
    publisher: 'Deccan Herald',
    title: 'Buses to run as usual on Monday, operators say',
    url: 'https://www.deccanherald.com/india/karnataka/transport-strike-status',
    published: '2026-09-26',
    kind: 'Local report',
    stance: 'contradicts',
    excerpt:
      'A private operators association said on Saturday that its members would operate as usual through next week, adding that negotiations were still under way. An office-bearer said that a decision on any withdrawal would require seven days notice, which had not been issued. Several operators said they had not received any advisory suggesting otherwise.',
  },
  strikeUnion: {
    id: 'src-strike-union',
    publisher: 'Transport Federation circular',
    title: 'Notice: discussion of common demands scheduled',
    url: 'https://example-transport-federation.org/notices',
    published: '2026-09-22',
    kind: 'Stakeholder statement',
    stance: 'unclear',
    excerpt:
      'A general body meeting of member associations will be held to consider the outcome of discussions with authorities regarding fare revision, permit fees, and welfare measures. A decision on the future course of action will be taken after the meeting. Members will be informed of any change to operating schedules in accordance with applicable notice requirements.',
  },
  wallPhoto: {
    id: 'src-wall-photo',
    publisher: 'Agence France-Presse',
    title: 'Reuters archive: image from 2019 floods recirculating',
    url: 'https://imagefactcheck.afp.com/archive-2019-flood-image',
    published: '2026-09-19',
    kind: 'Image verification',
    stance: 'contradicts',
    excerpt:
      'The photograph was taken in 2019 by a staff photographer and filed in the agency archive under the location and date recorded at the time of capture. The image carries intact archival metadata including capture date and camera information. Comparison with the visual record shows that the structure visible in the image has since been rebuilt twice, and that current photographs of the location do not match the scene.',
  },
  wallReport: {
    id: 'src-wall-report',
    publisher: 'Local district bulletin',
    title: 'Damaged section closed; repairs under way',
    url: 'https://district-bulletin.example.org/notice-repair',
    published: '2026-09-20',
    kind: 'Official bulletin',
    stance: 'unclear',
    excerpt:
      'Maintenance work on the approach is in progress and one lane remains closed to traffic while inspections are completed. Road users are advised to use the alternate route announced earlier this week. The bulletin does not reference the photograph circulating on social platforms.',
  },
  whoUpdate: {
    id: 'src-who-update',
    publisher: 'World Health Organization',
    title: 'Questions and answers: what the change in mpox status does and does not mean',
    url: 'https://www.who.int/news-room/questions-and-answers/item/mpox-status',
    published: '2023-05-16',
    kind: 'Primary explainer',
    stance: 'supports',
    excerpt:
      'Ending the public health emergency of international concern does not mean that the threat is over, and it does not mean that the response stops. Surveillance, laboratory testing, and vaccination for people at risk continue. The status is reviewed whenever the Emergency Committee advises the Director-General that circumstances may have changed, and the designation can be reinstated.',
  },
  ombudsman: {
    id: 'src-banking-ombudsman',
    publisher: 'Banking Ombudsman',
    title: 'Consumer advisory: prize, reward and benefit messages',
    url: 'https://banking-ombudsman.example.org/advisories/reward-messages',
    published: '2024-03-04',
    kind: 'Consumer advisory',
    stance: 'contradicts',
    excerpt:
      'Complaints about messages promising prizes, gold, or rewards in exchange for account details continued to rise this quarter. No licensed institution offers benefits in exchange for credentials, a card number, or a one-time password. Messages that create urgency — a countdown, a deadline, a limited number of beneficiaries — are a consistent feature of these campaigns and are not used by banks for genuine schemes.',
  },
  rtcNotice: {
    id: 'src-rtc-notice',
    publisher: 'Regional Transport Authority',
    title: 'Statement on scheduled services',
    url: 'https://rtc.example.org/statements/services',
    published: '2026-09-27',
    kind: 'Official statement',
    stance: 'unclear',
    excerpt:
      'No notice of withdrawal of services has been received by this authority. Operators are expected to maintain published schedules until such notice is given and acknowledged. Any change to scheduled services will be published here, and passengers will be informed through the usual channels. The authority is not in a position to comment on internal discussions within associations.',
  },
  photoArchive: {
    id: 'src-photo-archive',
    publisher: 'Photo archive record',
    title: 'Original filing record for the photograph',
    url: 'https://archive.example.org/records/image-2019-flood',
    published: '2019-08-14',
    kind: 'Archive record',
    stance: 'contradicts',
    excerpt:
      'Filed 14 August 2019. Location and date recorded at the time of capture. Subsequent records for the same location show the structure rebuilt in 2021 and again in 2024, with changes to the parapet and the approach visible in later photographs. The image has been requested eleven times since filing, most often in the last two years.',
  },
  railReport: {
    id: 'src-rail-report',
    publisher: 'National Disaster Management Authority',
    title: 'Seasonal bulletin: river levels and advisories',
    url: 'https://ndma.example.org/bulletins/seasonal',
    published: '2026-09-18',
    kind: 'Primary statement',
    stance: 'unclear',
    excerpt:
      'River levels at the monitored stations remained within the seasonal band through the week. No new evacuation advisories were issued. District administrations have been asked to complete pre-season inspections of embankments and report observations by the end of the month.',
  },
}

const FU = {
  why: { id: 'why', label: 'Why this verdict?', action: 'why' },
  evidence: { id: 'more', label: 'Show me more evidence', action: 'evidence' },
  another: { id: 'another', label: 'Check another claim', action: 'another' },
}

/* -------------------------------------------------------------------------
   Scenarios
   ------------------------------------------------------------------------- */
export const SCENARIOS = {
  health: {
    claim: 'The World Health Organization has declared an end to the global health emergency for mpox.',
    verdict: 'Real',
    confidence: 0.93,
    intro: 'Here’s what I found.',
    reasoning: [
      'The organisation named in the claim says so itself, in a statement published on its own website and dated the same day the decision was reported by the main wire services.',
      'The reasoning given is specific and checkable: a sustained fall in reported cases from the peak, no change in how severe the illness is, and improved access to testing and vaccines. Each of those is separately reported.',
      'Nothing credible found disagrees with the claim. The only qualifications come from the same organisations, and they concern what the change does and does not mean — not whether it happened.',
    ],
    evidence: [E.whoStatement, E.reutersMpox, E.apMpox],
    why: [
      'The claim turns on a single factual hinge: did the organisation change the classification, and did it say so publicly? Three independent accounts say yes, and the first of them is the organisation’s own statement.',
      'Where a claim attributes an action to a named body, that body’s own publication is the strongest available evidence. Here it exists, it is dated, and its language matches the claim closely enough that no interpretation is needed.',
      'The two wire reports were written independently and agree on the timing, the mechanism, and the committee process behind the decision. Agreement between accounts that did not copy each other is what raises confidence here — not the number of sources on its own.',
    ],
    more: {
      intro: 'I widened the search beyond the first results.',
      // The strongest source resurfaces, which is worth showing rather than hiding.
      evidence: [E.whoUpdate, E.reutersMpox],
      note: 'I looked for anything that disputes the claim and found nothing beyond commentary on what the change means in practice.',
    },
    followUps: [FU.why, FU.evidence, FU.another],
  },

  financial: {
    claim: 'The RBI is giving free gold coins to savings account holders who click a link and confirm their details.',
    verdict: 'Fake',
    confidence: 0.97,
    intro: 'This one is fabricated, and there is a paper trail.',
    reasoning: [
      'No such scheme exists. The central bank has published standing cautions saying it does not distribute prizes or benefits to individuals and does not ask for account details or one-time passwords.',
      'The claim traces back to forwarded messages rather than any announcement. The government fact-check unit examined the same wording and found the link led to a page imitating an official banking portal.',
      'The link’s domain was registered days before the messages began circulating, and it has no connection to any licensed bank.',
    ],
    evidence: [E.pibFactCheck, E.rbiPress, E.factChecker],
    why: [
      'The claim attributes a giveaway to a specific institution. That institution has publicly and repeatedly stated it does not do this, and its statement predates the message — so the claim cannot be a misreading of a genuine announcement.',
      'Independent examination found the mechanism described in the message, a link asking for account details and a one-time password, which is the pattern of a credential phishing page rather than a benefit scheme.',
      'Two of the three sources are direct statements from the bodies named in the claim. Where an institution contradicts a claim about itself, in writing, on its own website, that is close to conclusive.',
    ],
    more: {
      intro: 'I looked for the origin of the message.',
      evidence: [E.ombudsman, E.factChecker],
      note: 'The same wording appears across several languages with only the link changed, which is typical of a campaign rather than a single post.',
    },
    followUps: [FU.why, FU.evidence, FU.another],
  },

  local: {
    claim: 'Private buses in Bengaluru will not run on Monday because of a city-wide strike.',
    verdict: 'Unverified',
    confidence: 0.42,
    intro: 'The evidence here genuinely disagrees, so I’m not going to call it either way.',
    reasoning: [
      'Talks are real and are still going on. A federation has scheduled a general body meeting to decide on a course of action, and that meeting has not concluded.',
      'One local report quotes operators saying services will run as usual, and points out that any withdrawal requires seven days notice, which has not been issued.',
      'Another report quotes a federation office-bearer declining to confirm or deny a withdrawal. Those accounts do not line up, and the timing is close enough that a further announcement could still change the picture.',
    ],
    evidence: [E.strikeNews, E.strikePrint, E.strikeUnion],
    why: [
      'This claim is about something that has not happened yet. Evidence about the future is usually statements of intent, and intent can change between the statement and the event.',
      'The two local reports were published two days apart and describe the same negotiations differently: one emphasises that talks continue, the other that operators say services will run. Both can be true, and neither settles the claim.',
      'The notice requirement is the most concrete fact available — a seven-day notice has not been issued — but it establishes what is likely, not what is certain. That is why the reading sits below the middle rather than at one end.',
    ],
    more: {
      intro: 'I looked for the authority’s own statements.',
      evidence: [E.rtcNotice, E.strikeUnion],
      note: 'Nothing published by the federation or the transport authority confirms a withdrawal. If an announcement is made, this will be worth checking again.',
    },
    followUps: [FU.why, FU.evidence, FU.another],
  },

  image: {
    claim: 'This photograph shows the bridge that collapsed this week.',
    verdict: 'Fake',
    confidence: 0.88,
    intro: 'The photograph is real. It just isn’t from this week.',
    reasoning: [
      'The image carries archival metadata that places it in 2019, filed by a news agency under the location and date recorded at capture time.',
      'The structure in the photograph has been rebuilt twice since, and current photographs of the same location do not match the scene in the image.',
      'Repairs to the approach are genuinely under way, and one lane is closed. That real, smaller story is what the photograph has been attached to.',
    ],
    evidence: [E.wallPhoto, E.wallReport, E.railReport],
    why: [
      'For images, the useful question is rarely "is this photograph real?" — it usually is. The question is whether it shows what the caption says it shows, here and now.',
      'Three things were checked: the file’s own metadata, the visual record of the location over time, and the official notices about the structure. The metadata dates the picture; the visual record shows the location has changed since; the notices describe a lane closure rather than a collapse.',
      'That combination is why the reading is high but not at the top of the scale. Metadata can be stripped when a file is re-shared, so the dating rests on the archive record as well.',
    ],
    more: {
      intro: 'I looked for other uses of the same photograph.',
      evidence: [E.photoArchive, E.wallPhoto],
      note: 'The image has circulated with at least four different captions in the past two years, attached to different events.',
    },
    followUps: [FU.why, FU.evidence, FU.another],
  },

  general: {
    claim: null,
    verdict: 'Unverified',
    confidence: 0.34,
    intro: 'I couldn’t find enough to settle this.',
    reasoning: [
      'Nothing in the sources I can reach speaks directly to this claim. That is a limit of what I can see, not a finding about the claim itself.',
      'What I did find is adjacent rather than responsive — related material that does not confirm or contradict the specific thing you asked about.',
      'Unverified is the honest answer here. It means I don’t know, not that the claim is false.',
    ],
    evidence: [E.railReport],
    why: [
      'I looked for direct statements, reports, and any records matching the claim’s specifics. The searches returned either nothing or material about a different but similarly worded subject.',
      'A single adjacent source is not enough to raise a reading. Counting sources only helps when they address the claim; here, adding more of the same material would not change the answer.',
      'The most useful thing you can add is detail — a date, a place, who made the claim, or where you saw it. Any of those narrows the search considerably.',
    ],
    more: {
      intro: 'I checked the narrower angles.',
      evidence: [E.railReport],
      note: 'If you can tell me when or where you saw this, I can search with much more precision.',
    },
    followUps: [FU.why, FU.another],
  },
}

export const STARTERS = [
  {
    id: 'starter-health',
    scenario: 'health',
    label: 'Did the WHO end the mpox public health emergency?',
    detail: 'A claim about an official announcement',
  },
  {
    id: 'starter-financial',
    scenario: 'financial',
    label: 'The RBI is giving ₹5 lakh to savings account holders who click a link',
    detail: 'A forwarded message in a family group',
  },
  {
    id: 'starter-local',
    scenario: 'local',
    label: 'Private buses in Bengaluru are on strike on Monday',
    detail: 'Something you heard from a neighbour',
  },
]

/** Picks a scenario from the user's message. Keyword matching stands in for
 *  the real classification step the backend will do. */
export function detectScenario(text = '', image = null) {
  const q = text.toLowerCase()

  if (!q && image) return 'image'
  if (/(gold|rbi|coin|lakh|whatsapp|prize|luck ?y)/.test(q)) return 'financial'
  if (/(strike|bus|transport|bengaluru|bangalore|shut ?down)/.test(q)) return 'local'
  if (/(who|mpox|emergency|outbreak|vaccine|monkey ?pox)/.test(q)) return 'health'
  if (/(photo|picture|image|screenshot|this week|bridge)/.test(q)) return 'image'
  return 'general'
}
