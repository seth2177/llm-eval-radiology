"""The checker against impressions written the way radiologists and LLMs actually write them,
not the scripted model's templates. Each line is (detector findings, impression, expected errors)."""
import pytest

from llm_eval.cases import AIFinding
from llm_eval.checker import check, read

R6 = [AIFinding("right", 6.3, 0.9)]
L12 = [AIFinding("left", 12.4, 0.95)]
L4 = [AIFinding("left", 4.2, 0.8)]
NONE = []

CASES = [
    # faithful, varied phrasing
    (R6, "6 mm solid nodule in the right upper lobe. Recommend non-contrast chest CT in 6-12 months.", []),
    (R6, "Right lung nodule, 6.3 mm. Fleischner 2017: follow-up CT at 6 to 12 months.", []),
    (R6, "A 0.6 cm right pulmonary nodule is noted; CT follow-up in 6-12 months is recommended.", []),
    (R6, "Right pulmonary nodule measuring 7 x 6 mm. Follow-up CT in 6–12 months.", []),
    (L12, "12-mm left lower lobe nodule. Consider PET/CT or tissue sampling.", []),
    (L12, "Left lung nodule measuring 1.2 cm. Recommend CT in 3 months.", []),
    (L4, "4 mm left lung nodule. No routine follow-up is recommended.", []),
    (L4, "Tiny 4.2 mm nodule, left lung. Follow-up is not recommended.", []),
    (NONE, "No pulmonary nodule identified.", []),
    (NONE, "No suspicious pulmonary nodules.", []),
    (NONE, "Lungs are clear without pulmonary nodules. No follow-up imaging needed.", []),
    # unfaithful
    (R6, "Left lung nodule, 6 mm. Follow-up CT in 6-12 months.", ["LATERALITY"]),
    (R6, "Right lung nodule, 9 mm. Follow-up CT in 6-12 months.", ["SIZE"]),
    (R6, "Right lung nodule. Follow-up CT in 6-12 months.", ["SIZE_MISSING"]),
    (R6, "Right lung nodule, 6 mm.", ["FOLLOWUP_MISSING"]),
    (R6, "Right lung nodule, 6 mm. No routine follow-up is recommended.", ["FOLLOWUP_WRONG"]),
    (R6, "No pulmonary nodule identified.", ["OMISSION"]),
    (NONE, "Right lung nodule measuring 5 mm. No routine follow-up.", ["HALLUCINATION"]),
    (NONE, "No pulmonary nodule identified. Recommend follow-up CT in 6-12 months.", ["FOLLOWUP_UNWARRANTED"]),
    (R6, "Right lung nodule, 6 mm. Follow-up CT in 6-12 months. Additional 3 mm left lung nodule, "
         "no routine follow-up.", ["HALLUCINATION"]),
    (L12, "Left lung nodule, 1.2 cm. No routine follow-up, or CT in 3 months if high risk.", ["FOLLOWUP_WRONG"]),
    # cannot be read confidently: never passed silently
    (R6, "Findings as described above.", ["UNPARSEABLE"]),
    (R6, "Nodule measuring 6 mm. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Bilateral nodules, largest 6 mm.", ["UNPARSEABLE"]),

    # --- regressions from adversarial review: each of these used to pass silently or false-alarm ---
    # a negation in the same sentence must not hide a real claim
    (NONE, "Right lung nodule measuring 6 mm, no additional nodules identified.", ["HALLUCINATION"]),
    (NONE, "A 6 mm right lung nodule with no other nodules. Follow-up CT in 6-12 months.", ["HALLUCINATION"]),
    (R6, "6 mm right lung nodule without other nodules. Follow-up CT in 6-12 months.", []),
    (R6, "No calcified nodule; a noncalcified right lung nodule measures 6 mm. Follow-up CT 6-12 months.", []),
    # a negated recommendation is not a recommendation
    (L12, "Left lung nodule 12 mm. PET/CT is not indicated.", ["FOLLOWUP_WRONG"]),
    (L12, "Left lung nodule 12 mm. No biopsy needed.", ["FOLLOWUP_WRONG"]),
    (L12, "Left lung nodule 12 mm. No PET/CT needed; routine follow-up CT in 12 months.", ["FOLLOWUP_WRONG"]),
    (R6, "Right lung nodule 6 mm. Follow-up CT in 6-12 months is not required.", ["FOLLOWUP_WRONG"]),
    (R6, "6 mm right lung nodule. Follow-up CT in 6-12 months (no PET required).", []),
    (R6, "Right lung nodule, 6 mm. Recommend CT in 6-12 months. CT without contrast is sufficient.", []),
    # the size must be the nodule's, not a slice, a lymph node or a prior
    (R6, "A 6 mm right hilar lymph node adjacent to a 12 mm right lung nodule. Follow-up CT in 6-12 months.", ["SIZE"]),
    (R6, "On 6 mm reconstructions, a 12 mm right lung nodule. Follow-up CT in 6-12 months.", ["SIZE"]),
    (R6, "On 1 mm slices there is a right lung nodule measuring 6 mm. Follow-up CT in 6-12 months.", []),
    (R6, "Right lung nodule, 6 mm, adjacent to a 10 mm lymph node. Follow-up CT in 6-12 months.", []),
    (R6, "Right lung nodule less than 6 mm. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Right lung nodule measuring 5-6 mm. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Right lung nodule, 6,3 mm. Follow-up CT in 6-12 months.", []),
    (R6, "Right upper lobe nodule measuring 6 by 7 mm. Follow-up CT in 6-12 months.", []),
    # comparison language with no prior study is flagged, not trusted
    (R6, "Stable 6 mm right lung nodule. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (NONE, "No change in right nodule. No follow-up needed.", ["UNPARSEABLE"]),
    (R6, "Right lung nodule 6 mm, previously 12 mm. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Right lung nodule, 6 mm. Recommend CT in 6 to 12 months; if stable, consider CT at 18-24 months.", []),
    # follow-up intervals outside the guideline are wrong, not missing
    (L12, "Left lung nodule 12 mm. Recommend follow-up CT in 6 months.", ["FOLLOWUP_WRONG"]),
    (NONE, "No pulmonary nodule identified. Consider CT in 12 months.", ["FOLLOWUP_UNWARRANTED"]),
    (R6, "Right upper lobe nodule, 6mm. Follow-up CT 6-12mo.", []),
    (R6, "Right lung nodule 6 mm. Recommend a 6- to 12-month follow-up CT.", []),
    (R6, "Right upper lobe nodule, 6 millimeters. Follow-up CT in six to twelve months.", []),
    (L4, "4 mm left lung nodule. No further imaging is needed.", []),
    # more ways of writing it
    (R6, "RUL nodule, 6 mm. Follow-up CT in 6-12 months.", []),
    (R6, "LUL nodule, 6 mm. Follow-up CT in 6-12 months.", ["LATERALITY"]),
    (R6, "IMPRESSION:\n1. 6 mm right upper lobe nodule.\n2. Per Fleischner 2017, follow-up CT in 6-12 months.", []),
    (R6, "- Right lung nodule: 6 mm\n- Recommendation: CT in 6-12 months", []),
    (R6, "Right lung nodule. 6 mm. CT in 6-12 months.", []),
    (NONE, "Pulmonary nodules: none.", []),
    (NONE, "A right lung nodule is not seen.", []),
    (R6, "6 mm nodule abutting the left atrium in the right lower lobe. CT in 6-12 months.", ["UNPARSEABLE"]),

    # --- second review round ---
    (R6, "Right lung nodule 6 mm. Follow-up CT in 18-24 months.", ["FOLLOWUP_WRONG"]),          # skips 6-12
    (NONE, "No pulmonary nodule identified. Repeat chest CT is advised.", ["FOLLOWUP_UNWARRANTED"]),
    (NONE, "No pulmonary nodule identified. Follow-up CT in one year.", ["FOLLOWUP_UNWARRANTED"]),
    (L4, "4 mm left lung nodule. Recommend annual low-dose CT.", ["FOLLOWUP_WRONG"]),
    (L12, "Left lung nodule 12 mm. Biopsy can be deferred.", ["FOLLOWUP_WRONG"]),
    (L12, "Left lung nodule 12 mm. Rather than PET/CT, routine surveillance.", ["FOLLOWUP_WRONG"]),
    (R6, "Right lung nodule 6 mm. PET/CT: not indicated. Follow-up CT in 6-12 months.", []),
    (R6, "Right lung nodule 6 mm. For a patient with no risk factors CT in 6-12 months is advised.", []),
    (NONE, "Right lung nodule, not seen on soft tissue windows.", ["HALLUCINATION"]),
    (NONE, "There is no doubt a right nodule is present.", ["HALLUCINATION"]),
    (R6, "Right lung nodule. Mediastinal node 6 mm. CT in 6-12 months.", ["SIZE_MISSING"]),
    (R6, "Right lung nodule measuring 6 mm, lymph nodes are normal. CT in 6-12 months.", []),
    (R6, "6 mm thick-walled right lung nodule. CT in 6-12 months.", []),
    (R6, "Right lung nodule measures 6 mm or less. CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "6 mm right thyroid nodule. Follow-up CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Known 6 mm right lung nodule. CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "6 mm right lung nodule, again seen. CT in 6-12 months.", ["UNPARSEABLE"]),
    (NONE, "No new or suspicious pulmonary nodules.", []),
    (R6, "Right lung nodule 6 mm. Follow-up CT in 6-12 months to confirm stability.", []),
    (R6, "Right lung nodule 6 mm (6-8 mm category): CT in 6-12 months.", []),
]


@pytest.mark.parametrize("findings,text,expected", CASES, ids=[c[1][:40] for c in CASES])
def test_checker(findings, text, expected):
    assert check(text, findings).errors == expected


def test_decimal_point_is_not_a_sentence_break():
    r = read("Right lung nodule, 6.3 mm. Follow-up CT in 6-12 months.")
    assert len(r.claims) == 1 and r.claims[0].size_mm == pytest.approx(6.3)
    assert r.claims[0].followups == {"ct_6_12_months"}


def test_follow_up_attaches_to_the_nodule_it_follows():
    r = read("Right nodule 6 mm, CT in 6-12 months. Left nodule 12 mm. Consider PET/CT.")
    assert [c.side for c in r.claims] == ["right", "left"]
    assert r.claims[0].followups == {"ct_6_12_months"}
    assert r.claims[1].followups == {"ct_3_months_pet_or_sampling"}


# --- more than one detector finding: Fleischner's multiple-nodule table, scored per nodule ---------------

R6L12 = [AIFinding("right", 6.3, 0.9), AIFinding("left", 12.4, 0.9)]      # largest > 8 mm: CT in 3-6 months
RR = [AIFinding("right", 4.2, 0.8), AIFinding("right", 7.0, 0.9)]          # two on one side
SMALL = [AIFinding("right", 3.1, 0.7), AIFinding("left", 4.4, 0.8)]        # all < 6 mm: no routine follow-up
EDGE = [AIFinding("right", 5.5, 0.8), AIFinding("left", 3.0, 0.7)]         # 5.5 rounds to 6: CT in 3-6 months

MULTI = [
    # faithful
    (R6L12, "Right lung nodule, 6 mm. Left lung nodule, 12 mm. Recommend CT in 3-6 months, then consider CT at "
            "18-24 months.", []),
    (R6L12, "A 6 mm right lung nodule and a 12 mm left lung nodule. Follow-up CT in 3 to 6 months.", []),
    (R6L12, "Multiple pulmonary nodules. Right lung nodule, 6.3 mm. Left lower lobe nodule, 1.2 cm. CT in 3-6 months.",
     []),
    (R6L12, "Left lung nodule 12 mm. Right lung nodule 6 mm. For multiple nodules, CT in three to six months.", []),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. CT in 3-6 months; PET/CT is not needed.", []),
    (RR, "Right upper lobe nodule, 7 mm. Right lower lobe nodule, 4 mm. CT in 3-6 months.", []),
    (SMALL, "Right lung nodule 3 mm. Left lung nodule 4 mm. No routine follow-up is recommended.", []),
    (SMALL, "Right lung nodule 3 mm. Left lung nodule 4.4 mm. No follow-up needed for these nodules.", []),
    (EDGE, "Right lung nodule 5.5 mm. Left lung nodule 3 mm. CT in 3-6 months.", []),
    # the single-nodule table applied to multiple nodules is the classic mistake
    (R6L12, "Right lung nodule, 6 mm. Left lung nodule, 12 mm. Consider PET/CT.", ["FOLLOWUP_WRONG"]),
    (R6L12, "Right lung nodule, 6 mm. Left lung nodule, 12 mm. Consider CT in 3 months, PET/CT, or tissue sampling.",
     ["FOLLOWUP_WRONG"]),
    (RR, "Right lung nodule, 7 mm. Right lung nodule, 4 mm. Follow-up CT in 6-12 months.", ["FOLLOWUP_WRONG"]),
    (SMALL, "Right lung nodule 3 mm. Left lung nodule 4 mm. Recommend CT in 3-6 months.", ["FOLLOWUP_WRONG"]),
    (EDGE, "Right lung nodule 5.5 mm. Left lung nodule 3 mm. No routine follow-up is recommended.", ["FOLLOWUP_WRONG"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. Follow-up CT in 3-6 months is not required.",
     ["FOLLOWUP_WRONG"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. CT in 6-12 months, then 18-24 months.",
     ["FOLLOWUP_WRONG"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. CT in 18-24 months.", ["FOLLOWUP_WRONG"]),
    (R6L12, "Right lung nodule, 6 mm. Left lung nodule, 12 mm.", ["FOLLOWUP_MISSING"]),
    # per-nodule errors
    (R6L12, "Left lung nodule, 6 mm. Right lung nodule, 12 mm. CT in 3-6 months.", ["LATERALITY"]),      # swapped
    (R6L12, "Right lung nodule 6 mm. Right lung nodule 12 mm. CT in 3-6 months.", ["LATERALITY"]),
    (RR, "Right lung nodule, 7 mm. Left lung nodule, 4 mm. CT in 3-6 months.", ["LATERALITY"]),
    (R6L12, "Right lung nodule 9 mm. Left lung nodule 12 mm. CT in 3-6 months.", ["SIZE"]),
    (RR, "Right lung nodule, 7 mm. Right lung nodule, 7 mm. CT in 3-6 months.", ["SIZE"]),
    (R6L12, "Right lung nodule. Left lung nodule, 12 mm. CT in 3-6 months.", ["SIZE_MISSING"]),
    (R6L12, "Right lung nodule, 6 mm. CT in 3-6 months.", ["OMISSION"]),
    (R6L12, "No pulmonary nodule identified.", ["OMISSION"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. Additional 4 mm right lung nodule. CT in 3-6 months.",
     ["HALLUCINATION"]),
    (R6L12, "Left lung nodule, 12 mm. CT in 3 months.", ["FOLLOWUP_WRONG", "OMISSION"]),
    # can't be read one way only: never guessed
    (R6L12, "Bilateral pulmonary nodules, largest 12 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Right and left lung nodules measuring 6 and 12 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Right 6 mm and left 12 mm lung nodules. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Two pulmonary nodules: right lung 6 mm; left lung 12 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. The largest nodule warrants CT in 3-6 months.",
     ["UNPARSEABLE"]),
    (R6L12, "Stable right lung nodule 6 mm. Left lung nodule 12 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (RR, "Two right lung nodules measuring 4 mm and 7 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Multiple pulmonary nodules. CT in 3-6 months.", ["UNPARSEABLE"]),
    # found by hand while building this: each used to be a false alarm (OMISSION or HALLUCINATION)
    (RR, "Right lung nodules measuring 4 and 7 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (RR, "Right lung nodule 4 mm. Another 7 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Multiple pulmonary nodules, the largest in the left lung. Right lung nodule 6 mm. Left lung nodule 12 mm. "
            "CT in 3-6 months.", ["UNPARSEABLE"]),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm and several smaller nodules. CT in 3-6 months.",
     ["UNPARSEABLE"]),
    (RR, "Right lung nodule. 4 mm. Right lung nodule, 7 mm. CT in 3-6 months.", []),
    (R6L12, "1. Right lung nodule, 6 mm.\n2. Left lung nodule, 12 mm.\n3. CT in 3-6 months.", []),
    (R6L12, "Right lung nodule 6 mm; left lung nodule 12 mm; CT in 3-6 months.", []),
    (R6L12, "Right lung nodule 6 mm. Left lung nodule 12 mm. CT in 3-6 months or PET/CT.", ["FOLLOWUP_WRONG"]),
    # two readings, two different verdicts (the 9 mm one is wrong-sized, or the size-less one is unsized)
    (R6L12, "Right lung nodule. Right lung nodule, 9 mm. Left lung nodule, 12 mm. CT in 3-6 months.", ["UNPARSEABLE"]),
]


@pytest.mark.parametrize("findings,text,expected", MULTI, ids=[c[1][:40] for c in MULTI])
def test_checker_multiple(findings, text, expected):
    assert check(text, findings).errors == expected


@pytest.mark.parametrize("findings,text,expected", [
    # "and" splits a sentence only when each piece names one nodule
    (R6, "Right lung nodule 6 mm and no other nodules. CT in 6-12 months.", []),
    # (the follow-up attaches to the last nodule named, here the invented one; the report is flagged either way)
    (R6, "A 6 mm right lung nodule and a 4 mm left lung nodule. CT in 6-12 months.",
     ["FOLLOWUP_MISSING", "HALLUCINATION"]),
    # a side and a size with no nodule named is not dropped silently
    (R6, "Right lung nodule, 6 mm. Left lung 12 mm. CT in 6-12 months.", ["UNPARSEABLE"]),
    (R6, "Right lung nodule, 6 mm. Right hilar lymph node 10 mm. CT in 6-12 months.", []),
    # the multiple-nodule interval is wrong for a single nodule, with or without the 18-24 month scan
    (R6, "Right lung nodule 6 mm. CT in 3-6 months, then consider CT at 18-24 months.", ["FOLLOWUP_WRONG"]),
    (R6, "Right lung nodule 6 mm. For these nodules, CT in 6-12 months.", []),
], ids=lambda x: x[:40] if isinstance(x, str) else None)
def test_checker_changes_that_also_touch_single_nodules(findings, text, expected):
    assert check(text, findings).errors == expected


def test_follow_up_counts_once_for_the_study_wherever_it_is_written():
    r = check("Right lung nodule, 6 mm, CT in 3-6 months. Left lung nodule, 12 mm.", R6L12)
    assert r.errors == [] and r.reading.claims[0].followups == {"ct_3_6_months"}


def test_too_many_findings_are_out_of_scope():
    seven = [AIFinding("right" if i % 2 else "left", 3.0 + 1.5 * i, 0.9) for i in range(7)]
    assert check("Multiple pulmonary nodules.", seven).errors == ["OUT_OF_SCOPE"]
