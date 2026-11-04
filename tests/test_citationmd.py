import gzip
import json
from pathlib import Path

import pytest

from bialy import citationmd

CORPUS = json.loads(gzip.decompress((Path(__file__).parent / "fixtures/citation-grammar.json.gz").read_bytes()))
KNOWN = set(CORPUS["known_paths"])

# Accepted differences from the host parser, on shapes no training judgement rests on.
TABLE_CELLS = "markdown-it reads table cells apart, so a pipe in a quoted excerpt ends the cell"
SHORTCUT_REFERENCE = "goldmark takes an undefined following bracket or an open parenthesis as a shortcut reference"
DIVERGENCES = {
    '| a |\n|---|\n| [read#9 "a|b"] |': TABLE_CELLS,
    '[r]: https://unobs.test/d\n\n[r][read#9 "[l](https://unobs.test/e)"]': SHORTCUT_REFERENCE,
    '>[r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
    ' [r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
    '\n[r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
    '\n\n[r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
    '- [r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
    '> [r]: https://unobs.test/d\n\n[r](': SHORTCUT_REFERENCE,
}


def _occurrence(occurrence):
    fields = {"kind": occurrence.kind, "token": occurrence.token}
    if occurrence.citation:
        citation = occurrence.citation
        fields["citation"] = {key: value for key, value in (("handle", citation.handle), ("path", citation.path), ("line", citation.line),
                                                            ("end_line", citation.end_line), ("excerpt", citation.excerpt)) if value}
    for key in ("destination", "artifact", "method", "reason", "target", "url"):
        if getattr(occurrence, key):
            fields[key] = getattr(occurrence, key)
    return fields


def _issue(code, kind, token):
    # The block parser trims a line's trailing space before the scanner sees it.
    return {key: value for key, value in (("code", code), ("kind", kind), ("token", token.rstrip())) if value}


def _diverges(case):
    document = citationmd.parse(case["source"], known_path=KNOWN.__contains__ if case.get("known_paths") else None)
    got = [_occurrence(o) for o in document.occurrences], [_issue(i.code, i.kind, i.token) for i in document.issues]
    want = case["occurrences"], [_issue(i["code"], i.get("kind", ""), i.get("token", "")) for i in case["issues"]]
    return got != want


def test_kinds_match_the_host_registry():
    assert citationmd.KINDS == set(CORPUS["kinds"])


def test_the_port_answers_the_host_corpus_exactly():
    assert len(CORPUS["cases"]) > 5000
    diverging = {case["source"] for case in CORPUS["cases"] if _diverges(case)}
    assert diverging == set(DIVERGENCES)


@pytest.mark.parametrize("kind", [citationmd.CITATION, citationmd.WEB_LINK, citationmd.WORKSPACE_LINK, citationmd.ARTIFACT, citationmd.VERIFY, citationmd.REPAIR])
def test_the_corpus_exercises_every_occurrence_kind(kind):
    assert any(o["kind"] == kind for case in CORPUS["cases"] for o in case["occurrences"])


def test_the_corpus_exercises_the_issues_a_student_can_make():
    codes = {issue["code"] for case in CORPUS["cases"] for issue in case["issues"]}
    assert {"malformed_citation", "invalid_range", "malformed_artifact", "malformed_repair", "invalid_verification",
            "conflicting_verification", "unterminated_node"} <= codes


@pytest.mark.parametrize("text, fence", [
    ('Above.\n\n```json\n{"headline": "h"}\n```\n', '{"headline": "h"}\n'),
    ('```JSON\n{}\n```', '{}\n'),
    ('Above.\n\n```json\n{}', None),
    ('```python\n{}\n```', None),
    ('- list\n\n  ```json\n  {}\n  ```', None),
    ('```json\n{}\n```\n\nBelow.', None),
    ('Plain.', None),
])
def test_report_fence_is_a_closed_top_level_json_fence_ending_the_document(text, fence):
    assert citationmd.parse(text).report_fence == fence


def test_over_budget_source_is_one_issue_and_no_citation():
    document = citationmd.parse("[read#1] " * (citationmd.MAX_SOURCE_BYTES // 9 + 1))
    assert [issue.code for issue in document.issues] == ["source_budget_exceeded"] and not document.occurrences


def test_occurrences_past_the_budget_are_counted():
    document = citationmd.parse("[read#1] " * (citationmd.MAX_OCCURRENCES + 3))
    assert len(document.occurrences) == citationmd.MAX_OCCURRENCES
    assert [(issue.code, issue.token) for issue in document.issues] == [("occurrence_budget_exceeded", "3")]


def test_a_node_past_the_cap_is_too_long_on_its_own_line():
    document = citationmd.parse("[read#1 \"%s\"]\n[read#2]" % ("x" * citationmd.MAX_NODE_BYTES))
    assert [issue.code for issue in document.issues] == ["node_too_long"]
    assert [o.token for o in document.occurrences] == ["read#2"]


def test_handles_name_registered_kinds():
    assert citationmd.is_handle("read#1") and citationmd.is_handle("leg-a:read#1")
    assert not any(citationmd.is_handle(text) for text in ("read", "pr#1", "a.go", 3, None))
