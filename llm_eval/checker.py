"""Layer 2: is the impression faithful to what the detector reported?

The checker reads the impression text, pulls out its claims (each nodule's side and
size, and the follow-up attached to it) and compares them with the detector's
findings. It is rule-based and deterministic: the same text always gets the same
verdict, and every verdict can be traced to a line of code.

When the text is ambiguous (two sides or two sizes in one sentence, a qualified size
such as "less than 6 mm", comparison language with no prior study) it returns
UNPARSEABLE for human review rather than guessing. That is a design rule, not a
guarantee: the phrasings it has been tested against are in tests/test_checker.py.

Error codes
  HALLUCINATION         a nodule in the report that the detector did not report
  OMISSION              a detector finding missing from the report
  LATERALITY            right/left swapped
  SIZE                  size differs from the detector's by more than rounding (0.5 mm)
  SIZE_MISSING          a nodule reported without a size
  FOLLOWUP_WRONG        follow-up does not match Fleischner 2017 for the detector's size
  FOLLOWUP_MISSING      no follow-up recommendation for a nodule
  FOLLOWUP_UNWARRANTED  imaging follow-up recommended in a report with no nodule
  UNPARSEABLE           the checker could not tell what the report says
  OUT_OF_SCOPE          more than one detector finding (Fleischner's multiple-nodule table
                        is not implemented, so these are not scored)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import fleischner as F
from .cases import AIFinding

HALLUCINATION, OMISSION, LATERALITY, SIZE, SIZE_MISSING = (
    "HALLUCINATION", "OMISSION", "LATERALITY", "SIZE", "SIZE_MISSING")
FOLLOWUP_WRONG, FOLLOWUP_MISSING, FOLLOWUP_UNWARRANTED = "FOLLOWUP_WRONG", "FOLLOWUP_MISSING", "FOLLOWUP_UNWARRANTED"
UNPARSEABLE, OUT_OF_SCOPE = "UNPARSEABLE", "OUT_OF_SCOPE"
OTHER_INTERVAL = "other_interval"      # a follow-up interval that is not a Fleischner category

SIZE_TOLERANCE_MM = 0.5

# A period ends a sentence unless it is a decimal point (6.3 mm).
_SENTENCE = re.compile(r"(?<!\d)\.(?!\d)|\.(?=\s)|[;\n•]")
_NODULE = re.compile(r"\b(?:micro)?nodul(?:e|es|ar)\b")
# "no / without / negative for ... nodule(s)". Not "no more than", "no doubt", "no change" (a comparison).
_NEGATED = re.compile(r"\b(?:no(?!\s+(?:more|less|doubt|change|significant|interval)\b)|without|negative\s+for)\b"
                      r"(?:\s+[\w-]+){0,3}?\s+(?:pulmonary\s+)?nodul(?:e|es)\b")
# The nodule itself is the subject and the sentence ends there: "a right lung nodule is not seen."
_NOT_SEEN = re.compile(r"\bnodul\w*\s+(?:is\s+|are\s+)?not\s+(?:seen|identified|present|visuali[sz]ed|demonstrated)\s*$"
                       r"|\bnodul\w*\s*:\s*none\b")
_COMPARISON = re.compile(
    r"\b(?:stable|unchanged|no\s+(?:significant\s+|interval\s+)?change|changed|new|newly|prior|previous(?:ly)?|"
    r"interval\s+(?:change|growth|increase|decrease|development|enlargement)|compared|comparison|"
    r"increas\w*\s+(?:in\s+)?size|decreas\w*\s+(?:in\s+)?size|grown|growing|enlarg\w*|resolved|redemonstrat\w*|"
    r"known|persistent|persists?|again\s+(?:seen|noted|demonstrated|identified)|similar|remains?|larger|smaller|"
    r"as\s+before|since|no\s+longer)\b")
_OTHER_ORGAN = re.compile(r"\b(?:thyroid|breast|renal|kidney|adrenal|liver|hepatic|skin|subcutaneous|splenic)\b")
_SIDE = re.compile(r"\b(right|left|bilateral)\b")
_LOBES = ((re.compile(r"\b(?:rul|rml|rll)\b"), "right"), (re.compile(r"\b(?:lul|lll|lingula(?:r)?)\b"), "left"))
_SIZE = re.compile(
    r"(?P<q>(?:<|>|≤|≥|less\s+than|greater\s+than|up\s+to|at\s+least|at\s+most|under|over|sub-?)\s*)?"
    r"(?P<a>\d+(?:\.\d+)?)\s*(?:(?P<rng>-|to)\s*(?P<a2>\d+(?:\.\d+)?)\s*)?"
    r"(?:(?:mm\s*)?(?:[x×]|by)\s*(?P<b>\d+(?:\.\d+)?)\s*(?:(?:mm\s*)?[x×]\s*\d+(?:\.\d+)?\s*)?)?"
    r"-?\s*(?P<u>mm|millimet(?:er|re)s?|cm)\b(?P<post>\s+or\s+(?:less|smaller|more|larger|greater))?")
_SIZE_NOT_NODULE_BEFORE = re.compile(r"(?:previous\w*|prior|formerly|slice\s+thickness\s+of|reconstructed\s+at)\W*$")
# A structure word right after the number ("6 mm slices"), or a node later in the same clause
# ("6 mm right hilar lymph node"), means the size isn't the nodule's.
_SIZE_NOT_NODULE_AFTER = re.compile(r"^\s*(?:slices?|sections?|reconstruct\w*|(?:slice\s+)?thickness|collimation)\b"
                                    r"|^[^,;:()]{0,25}?\b(?:lymph|nodes?)\b")
_SIZE_ONLY = re.compile(r"^\W*(?:size|measur\w*|it\s+measures)?\W*\d+(?:\.\d+)?\s*(?:mm|cm)\W*$")
_CATEGORY_NOTE = re.compile(r"\((?:[^)]*(?:category|fleischner|<|>|\d\s*-\s*\d+\s*mm)[^)]*)\)")

_NONE = re.compile(r"\bno\s+(?:routine\s+|further\s+|imaging\s+|dedicated\s+)*follow[- ]?up\b|"
                   r"\bno\s+(?:further\s+)?imaging\s+(?:is\s+)?(?:needed|required|necessary|recommended)\b|"
                   r"(?:follow[- ]?up|further\s+imaging)\s+(?:imaging\s+)?is\s+not\s+"
                   r"(?:recommended|needed|required|necessary)")
_INTERVAL = re.compile(r"\b(?P<a>\d+)\s*-?\s*(?:(?:-|to)\s*(?P<b>\d+)\s*-?\s*)?(?P<u>months?|mos?|years?|yrs?)\b")
_PET_OR_SAMPLING = re.compile(r"\bpet\b|\btissue\s+sampling\b|\bbiopsy\b")
# An imaging recommendation with no interval ("repeat chest CT is advised"): not a Fleischner category.
_IMAGING_REC = re.compile(r"\b(?:follow[- ]?up|repeat|short[- ]term|short[- ]interval|surveillance|annual)\b[^.]*"
                          r"\b(?:ct|imaging|scan)\b|\b(?:ct|imaging)\b[^.]*\b(?:recommended|advised|suggested)\b")
_NEG_BEFORE = re.compile(r"\b(?:no|not|nor|avoid\w*)\b|\brather\s+than\b"
                         r"|\bwithout\b(?!\s+(?:iv\s+|intravenous\s+)?contrast)")
_NEG_AFTER = re.compile(r"\b(?:not|unnecessary|unwarranted|defer\w*|unlikely|premature|limited\s+value)\b")
_CLAUSE = re.compile(r"[,;:()]")
_CLAUSE_AFTER = re.compile(r"[,;()]")          # a colon doesn't end it: "PET/CT: not indicated"
THEN_18_24 = "then_18_24_months"


@dataclass
class Claim:
    side: str | None
    size_mm: float | None
    followups: set[str] = field(default_factory=set)
    negated_followups: set[str] = field(default_factory=set)
    text: str = ""


@dataclass
class Reading:
    negative_statement: bool
    claims: list[Claim]
    orphan_followups: set[str]
    orphan_negated: set[str]
    ambiguous: list[str]           # reasons the text can't be scored confidently
    sentences: list[str]


_NUMBER_WORDS = {"one": "1", "three": "3", "six": "6", "eight": "8", "twelve": "12", "eighteen": "18",
                 "twenty-four": "24"}


def _normalize(text: str) -> str:
    t = text.lower().replace("–", "-").replace("—", "-")
    t = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: _NUMBER_WORDS[m.group(1)], t)
    t = re.sub(r"\bif\s+stable\b", "if confirmed", t)           # a conditional, not a comparison
    t = re.sub(r"\ba\s+year\b", "1 year", t)
    t = re.sub(r"\bannual(?:ly)?\b", "annual 12 months", t)
    t = _CATEGORY_NOTE.sub(" ", t)                                  # "(6-8 mm category)" echoes the prompt
    t = re.sub(r"(\d),(\d)(?=\s*(?:mm|cm))", r"\1.\2", t)           # decimal comma: 6,3 mm
    for rx, side in _LOBES:
        t = rx.sub(side, t)
    return t


def _sizes(s: str, why: list[str]) -> list[float]:
    out = []
    for m in _SIZE.finditer(s):
        before, after = s[max(0, m.start() - 30):m.start()], s[m.end():m.end() + 25]
        if _SIZE_NOT_NODULE_BEFORE.search(before) or _SIZE_NOT_NODULE_AFTER.search(after):
            continue                                              # a slice, a lymph node, a prior size
        if m.group("q") or m.group("rng") or m.group("post"):
            why.append(f"qualified or ranged size: '{m.group(0).strip()}'")
            continue
        a = float(m.group("a"))
        mm = (a + float(m.group("b"))) / 2 if m.group("b") else a     # Fleischner: mean of long and short axis
        out.append(mm * 10 if m.group("u") == "cm" else mm)
    return out


def _followups(s: str) -> tuple[set[str], set[str]]:
    """(recommended categories, negated categories) in one sentence."""
    hits: list[tuple[str, int, int]] = []
    for m in _INTERVAL.finditer(s):
        a, b, unit = int(m.group("a")), m.group("b"), m.group("u")
        month = unit.startswith("mo")
        if month and a == 6 and b == "12":
            cat = F.CT_6_12
        elif month and a == 18 and b == "24":
            cat = THEN_18_24                   # valid only after a 6-12 month scan; check() decides
        elif month and a == 3 and b is None:
            cat = F.CT_3_PET
        else:
            cat = OTHER_INTERVAL
        hits.append((cat, m.start(), m.end()))
    hits += [(F.CT_3_PET, m.start(), m.end()) for m in _PET_OR_SAMPLING.finditer(s)]

    found, negated = set(), set()
    for cat, start, end in hits:
        cuts = [m.end() for m in _CLAUSE.finditer(s, 0, start)]
        before = " ".join(s[cuts[-1] if cuts else 0:start].split()[-4:])     # the few words just before
        nxt = _CLAUSE_AFTER.search(s, end)
        after = s[end:nxt.start() if nxt else len(s)]
        (negated if _NEG_BEFORE.search(before) or _NEG_AFTER.search(after) else found).add(cat)
    if _NONE.search(s):
        found.add(F.NONE)
    elif not hits and _IMAGING_REC.search(s) and not _NEG_AFTER.search(s):
        found.add(OTHER_INTERVAL)
    return found, negated


def read(text: str) -> Reading:
    t = _normalize(text)
    sentences = [s.strip() for s in _SENTENCE.split(t) if s and s.strip()]
    claims: list[Claim] = []
    orphan, orphan_neg, why = set(), set(), []
    negative = False
    for s in sentences:
        positive_part = _NEGATED.sub(" ", s)
        if positive_part != s:
            negative = True
        if _COMPARISON.search(positive_part):
            why.append(f"comparison language with no prior study: '{s}'")
        if _NOT_SEEN.search(positive_part) and not _SIZE.search(positive_part):
            negative, positive_part = True, ""
        fu, neg = _followups(positive_part)
        if _NODULE.search(positive_part) and _OTHER_ORGAN.search(positive_part):
            why.append(f"a nodule outside the lung: '{s}'")
            continue
        if _NODULE.search(positive_part):
            sides = set(_SIDE.findall(positive_part))
            sizes = sorted(set(round(x, 1) for x in _sizes(positive_part, why)))
            if len(sides) > 1 or "bilateral" in sides:
                why.append(f"more than one side in one sentence: '{s}'")
            if len(sizes) > 1:
                why.append(f"more than one size in one sentence: '{s}'")
            claims.append(Claim(next(iter(sides)) if len(sides) == 1 else None,
                                sizes[0] if len(sizes) == 1 else None, fu, neg, s))
            continue
        if claims:
            claims[-1].followups |= fu
            claims[-1].negated_followups |= neg
            if claims[-1].size_mm is None and _SIZE_ONLY.match(positive_part):   # "Right lung nodule. 6 mm."
                extra = _sizes(positive_part, why)
                if len(set(extra)) == 1:
                    claims[-1].size_mm = extra[0]
        else:
            orphan |= fu
            orphan_neg |= neg
    return Reading(negative, claims, orphan, orphan_neg, why, sentences)


@dataclass
class Verdict:
    errors: list[str]
    reading: Reading

    @property
    def ok(self) -> bool:
        return not self.errors


def check(text: str, findings: list[AIFinding]) -> Verdict:
    r = read(text)
    if len(findings) > 1:
        return Verdict([OUT_OF_SCOPE], r)
    if r.ambiguous or (not r.claims and not r.negative_statement) or any(c.side is None for c in r.claims):
        return Verdict([UNPARSEABLE], r)

    errors: list[str] = []
    if not findings:
        if r.claims:
            errors.append(HALLUCINATION)
        elif r.orphan_followups - {F.NONE}:
            errors.append(FOLLOWUP_UNWARRANTED)   # includes "repeat CT is advised" with no interval
        return Verdict(errors, r)

    f = findings[0]
    size = round(f.diameter_mm, 1)                 # the value the LLM was shown
    same = [c for c in r.claims if c.side == f.laterality]
    if same:
        c = min(same, key=lambda c: abs((c.size_mm or 1e9) - size))
    elif r.claims:
        errors.append(LATERALITY)
        c = min(r.claims, key=lambda c: abs((c.size_mm or 1e9) - size))
    else:
        return Verdict([OMISSION], r)

    if c.size_mm is None:
        errors.append(SIZE_MISSING)
    elif abs(c.size_mm - size) > SIZE_TOLERANCE_MM + 1e-9:
        errors.append(SIZE)

    expected = F.category(size)
    fu = c.followups | r.orphan_followups
    negated = c.negated_followups | r.orphan_negated
    if THEN_18_24 in fu:                          # "then 18-24 months" only counts after a 6-12 month scan
        fu = (fu - {THEN_18_24}) | (set() if F.CT_6_12 in fu else {OTHER_INTERVAL})
    negated.discard(THEN_18_24)
    if expected in negated:
        errors.append(FOLLOWUP_WRONG)             # "PET/CT is not indicated" for a 13 mm nodule
    elif not fu:
        errors.append(FOLLOWUP_MISSING)
    elif fu != {expected}:
        errors.append(FOLLOWUP_WRONG)             # wrong category, an unlisted interval, or a hedge naming two

    if len(r.claims) > 1:
        errors.append(HALLUCINATION)
    return Verdict(sorted(set(errors)), r)
