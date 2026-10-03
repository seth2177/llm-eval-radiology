# Method

## Population

Every `truth/<uid>.json` is a study. A study with no matching `results/<uid>.json` was not sent to a model
(dicom-ai-router matched no routing rule, e.g. head CT) and is counted but not scored. QA phantoms routed to
`ct-qa` are counted and excluded. The scored population is chest CTs that the `lung-nodule` model saw.

## Layer 1: detector outcome

Each synthetic study carries at most one nodule. The detector can still report more than one.

| Outcome | Truth | Detector |
|---|---|---|
| TP | nodule | nodule on the same side |
| WRONG_SIDE | nodule | nodule(s) only on the other side |
| FN | nodule | nothing |
| FP | no nodule | nodule |
| TN | no nodule | nothing |

A study where the detector reports the true nodule plus one it invented is TP for sensitivity; the invented one
counts against it in attribution (below).

Sensitivity = TP / (TP + FN + WRONG_SIDE). Specificity = TN / (TN + FP). Intervals are 95% Wilson score
intervals, which stay honest at the small n typical of per-stratum counts (0/6 is reported as 0-39%, not 0%).

Size error = detector diameter minus true diameter, on true positives only. Strata use the true density and
the Fleischner size bands (< 6, 6-8, > 8 mm, after rounding to the nearest mm).

## Layer 2: faithfulness

The LLM sees only the detector's findings, so it is scored against those, not against truth. Grading an LLM
against truth would blame it for the detector's misses.

The checker lowercases the impression, maps lobe abbreviations to a side (RUL, RML, RLL to right; LUL, LLL and
lingula to left), turns number words and decimal commas into digits, and splits it into sentences (a period
between digits is a decimal point). Then, for each sentence:

1. **Negations are removed before anything else is read.** "No / without / negative for ... nodule(s)" spans are
   cut out, and what is left is read again. "Right lung nodule, 6 mm, no additional nodules" is still a claim.
   "A right lung nodule is not seen" and "nodules: none" are negative statements.
2. **A remaining mention of a nodule is a claim.** A sentence with "and" between two nodules ("a 6 mm right lung
   nodule and a 12 mm left lung nodule") is read as two sentences, but only when each piece names exactly one
   nodule. A plural reference with no side and no size ("for these nodules," "multiple pulmonary nodules") points
   at nodules described elsewhere and is not a claim. Its side must be exactly one of right/left. Its size is the one
   measurement in the sentence that belongs to it: sizes attached to slices, sections, reconstructions, lymph
   nodes or a prior value are skipped. `A x B mm` is averaged, as Fleischner specifies. A size-only sentence
   right after a claim ("Right lung nodule. 6 mm.") attaches to it.
3. **Follow-up phrases attach to the claim they follow.** They map to the Fleischner categories: no routine
   follow-up / CT in 6-12 months (with an optional "then 18-24 months") / CT in 3 months, PET/CT or tissue
   sampling / CT in 3-6 months (the multiple-nodule interval, also with an optional "then 18-24 months"). Any
   other interval ("12 months," "6 months," "3-6 months") is recorded as a non-guideline
   interval, and so is an imaging recommendation with no interval at all ("repeat chest CT is advised").
   "18-24 months" counts only after a 6-12 or 3-6 month scan. A phrase negated in its own clause ("PET/CT is not
   indicated," "no biopsy needed," "biopsy can be deferred," "rather than PET/CT") is a negated
   recommendation, not a recommendation.

A report is `UNPARSEABLE` when a sentence names two sides or two sizes, a nodule sentence says "nodules" (it
describes more than one), a size appears with no nodule named ("left lung 12 mm," "another 7 mm"), a size is
qualified or given as a range ("less than 6 mm," "5-6 mm," "up to 7 mm"), comparison language appears with no prior study ("stable,"
"unchanged," "previously," "new," "known," "persistent," "again seen"), a nodule is in another organ (thyroid,
breast, kidney), or there is neither a claim nor a negative statement.

Scoring, with one detector finding:

- No claim on the finding's side: a claim on the other side is `LATERALITY`; no claim at all is `OMISSION`.
- Size must match the value the LLM was shown within 0.5 mm, which allows "6.3 mm" to be written "6 mm."
- Follow-up: if the correct category was negated, `FOLLOWUP_WRONG`. With none given, `FOLLOWUP_MISSING`. Any set
  other than exactly the correct category (a wrong one, a non-guideline interval, or a hedge naming two) is
  `FOLLOWUP_WRONG`.
- A second claim is a `HALLUCINATION`.

With no detector findings, any claim is a `HALLUCINATION`, and any imaging recommendation other than "no
follow-up" is `FOLLOWUP_UNWARRANTED`.

### More than one detector finding

Fleischner 2017 manages multiple solid nodules by the most suspicious one. The detector reports only side and
size, so that is the largest, and the table is short (`fleischner.multiple_category`):

| Largest nodule | Recommendation |
|---|---|
| < 6 mm | no routine follow-up |
| 6-8 mm | CT in 3-6 months, then consider CT at 18-24 months |
| > 8 mm | CT in 3-6 months, then consider CT at 18-24 months |

PET/CT or tissue sampling is not in the multiple table, so the single-nodule recommendation for a > 8 mm
nodule ("consider PET/CT") is `FOLLOWUP_WRONG` when there are other nodules. That is the mistake this table most
needs to catch. The multiple subsolid table is not implemented: the detector reports no density.

Each claim is matched to a finding. Every way of pairing as many findings with claims as possible is tried,
and the best one wins: first the most exact pairs (side and size within 0.5 mm), then the most pairs whose size
matches, then the most whose side matches. A matching size outranks a matching side because two nodules on one
side are common and two within 0.5 mm of each other are not; this is what lets a report that swaps the two
sides read as `LATERALITY` instead of two `SIZE` errors. In the winning pairing a pair on the wrong side is
`LATERALITY`, a pair whose size is off is `SIZE` (or `SIZE_MISSING`), an unpaired finding is `OMISSION` and an
unpaired claim is `HALLUCINATION`.

If two equally good pairings give different errors, the report can be read two ways and is `UNPARSEABLE`.
Example: findings right 6.3 mm and left 12.4 mm, report "Right lung nodule. Right lung nodule, 9 mm. Left lung
nodule, 12 mm." Either the unsized one is the 6.3 mm nodule (`SIZE_MISSING`, then the 9 mm one is invented) or
the 9 mm one is (`SIZE`, then the unsized one is invented). The checker does not pick.

Follow-up is scored once per study, against the multiple table and the sizes the LLM was shown, wherever in
the report it was written. With more than six detector findings the case is `OUT_OF_SCOPE` and not scored.

## Attribution

The detector is right when it is TN, or TP with no extra findings and a measured size in the same Fleischner
category as the true size and density. An extra finding is a nodule that isn't there, and a faithful report
will repeat it; it also moves the patient onto the multiple-nodule table. A 5 mm nodule measured as 6.3 mm is on the right side, but it moves the patient from no
follow-up to CT in 6-12 months, so it counts as a detector error. The LLM is faithful when the checker finds no
error. The 2x2 cross gives report right / LLM error / detector error / both wrong. Model errors and
out-of-scope cases are reported separately and left out of the faithfulness rate.

## Checker meta-evaluation

The scripted model plants at most one error per report, drawn from HALLUCINATION, OMISSION, LATERALITY, SIZE,
FOLLOWUP_WRONG and FOLLOWUP_MISSING (on detector-negative studies only HALLUCINATION is possible). Size errors
include ones just past the tolerance (0.6 and 1.0 mm). Wrong follow-ups include negated correct ones ("PET/CT
and tissue sampling are not indicated") and non-guideline intervals. Clean reports use the same traps in
harmless form ("no additional nodules identified," "PET/CT is not needed" next to the right recommendation). It records
what it planted; the checker never sees that record. Exact agreement means the checker's error set equals the
planted set, including the empty set on clean reports. `selftest` requires 100% at planted-error rates of 1.0
and 0.0 over three seeds.

The router data has no study where the detector reported two nodules, so `selftest` also scores multi-nodule
variants of it (`synthetic.py`): for each study with one detector finding, a copy with one or two more,
findings at least 1.5 mm apart, about 40% of them with every nodule under 6 mm (including 5.4 and 5.5 mm, the
rounding boundary). Multi-nodule reports are written one sentence per nodule or two joined by "and," sometimes
opened with "Multiple pulmonary nodules," with one recommendation for the study. A planted error touches one
nodule or the recommendation: one nodule dropped, moved to the other side, resized (never to within 0.5 mm of
another finding, so the planted report is never equivalent to a correct one), an invented extra nodule, the
recommendation left out, or a wrong one, including the single-nodule recommendation for the largest nodule
and a negated correct one ("Follow-up CT in 3-6 months is not required"). Outside CI, 300 seeds at planted-error
rates of 1.0, 0.5 and 0.0 (148,500 reports, 51,300 of them multi-nodule) gave no disagreement.

That the planted errors are found does not make the matching rule clinically right. It is a reading of an
impression, written down so it can be argued with: "the size decides which nodule a sentence means, before the
side does."

This proves the checker catches the error types it was built for, in the phrasings it was tested on. It does
not prove it catches every way a real LLM can be wrong; that is what UNPARSEABLE and human review of
`cases.jsonl` are for.
