"""The inline citation grammar of Painted Wolf Code agent Markdown.

A port of the host's parser, pinned by tests/fixtures/citation-grammar.json.gz, which
the host parser writes over its own corpus. It reports syntax only: nothing here says
whether a reference resolves. Parses run with no observed paths unless the caller has
some, so a bare `[Dockerfile]` is prose without them. The host's structural work
budgets are not ported, since they guard a long-lived process against hostile input.
"""

import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

from markdown_it import MarkdownIt

KINDS = frozenset({
    "command", "delete", "diff", "edit", "find", "git", "git_receipt", "grep", "http_response", "list", "overlay_receipt",
    "page", "page_geometry", "provenance", "read", "recall", "render", "replace", "restore", "rewrite", "scan",
    "secret_lifecycle", "skill", "stat", "summarize", "survey", "task", "terminal_session", "tui", "wc", "web", "write",
})
MAX_NODE_BYTES = 4096
MAX_SOURCE_BYTES = 2 << 20
MAX_OCCURRENCES = 1024
MAX_LINE = 1_000_000_000
MAX_REASON_BYTES = 2000
MAX_ISSUE_TOKEN_BYTES = 256
VERIFY_METHODS = {"inspection", "targeted", "project", "blocked"}
VERIFY_SEPARATORS = ("--", "-", "—", "–", ":")

CITATION, WEB_LINK, WORKSPACE_LINK, ARTIFACT, VERIFY, REPAIR = "citation", "web_link", "workspace_link", "artifact", "verify", "repair"

_HANDLE = r"(?:[a-zA-Z0-9_-]+:)*[a-z_]+#[1-9][0-9]*"
_HANDLE_KIND = re.compile(r"[a-z_]+\Z")
_HANDLE_WITH_COORDINATE = re.compile(r"(%s)(?::([0-9]+)(?:-([0-9]+))?)?\Z" % _HANDLE)
_COORDINATE_TAIL = re.compile(r":([0-9]+)(?:-([0-9]+))?\Z")
_ARTIFACT_ID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
_REPAIR_TARGET = re.compile(r"c[1-9][0-9]*\Z")
_RESERVED_HEAD = re.compile(r"\[(verify|artifact)[ \t]*:")
_REPAIR_HEAD = re.compile(r"\[cite[ \t]*:[ \t]*\S+[ \t]*=>")
_NAMESPACED_HEAD = re.compile(r"\[(?:verify|artifact|cite):([a-z_]+)#[1-9][0-9]*(?:[\]:, \t]|\Z)")
_MALFORMED_COORDINATE = re.compile(r"[^:#]*[./][^:#]*(?::[0-9]+:[0-9].*|:L[0-9].*|:[0-9]+-\Z|:[0-9]+-[^0-9].*|#L[0-9]*(?:-L?[0-9]*)?)\Z", re.S)
_HANDLE_SHAPED = re.compile(r"[a-zA-Z0-9_:-]+#[a-zA-Z0-9_-]*\Z")
_WEB_SCHEME = re.compile(r"[ \t]*https?:", re.I)
_ANY_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*:")
_LINE_FRAGMENT = re.compile(r"L([0-9]+)(?:-L?([0-9]+))?\Z")
_BARE_PREFIXES = ("https://", "http://", "ftp://", "www.")
_PATH_PUNCTUATION = set("-_.@+~%=/")
_LEXEME_BYTES = set("-_.@+~%=/:")
_UNICODE_SPACE = set("\t\n\v\f\r \x85\xa0                　﻿")

_PENDING, _RELEASE, _CLOSED, _UNTERMINATED, _OVERFLOW = range(5)
_LEXEME, _CITATION, _VERIFY, _ARTIFACT, _REPAIR = range(5)
_MODE_KIND = {_VERIFY: VERIFY, _ARTIFACT: ARTIFACT, _REPAIR: REPAIR}


@dataclass
class Citation:
    handle: str = ""
    path: str = ""
    line: int = 0
    end_line: int = 0
    excerpt: str = ""


@dataclass
class Occurrence:
    kind: str
    token: str
    citation: Citation | None = None
    destination: str = ""
    artifact: str = ""
    method: str = ""
    reason: str = ""
    target: str = ""
    url: str = ""


@dataclass
class Issue:
    code: str
    kind: str = ""
    token: str = ""


@dataclass
class Document:
    occurrences: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    report_fence: str | None = None

    def of(self, *kinds):
        return [o for o in self.occurrences if o.kind in kinds]


def is_handle(text):
    """Whether text is one evidence handle of a registered kind."""
    match = _HANDLE_WITH_COORDINATE.match(text) if isinstance(text, str) else None
    return bool(match) and _known_kind(_terminal_kind(match.group(1)))


def _bounded(token):
    raw = token.encode("utf-8")
    if len(raw) <= MAX_ISSUE_TOKEN_BYTES:
        return token
    return raw[:MAX_ISSUE_TOKEN_BYTES].decode("utf-8", "ignore")


def _known_kind(kind):
    return bool(_HANDLE_KIND.match(kind)) and kind in KINDS


def _is_web(destination):
    return bool(_WEB_SCHEME.match(destination))


def _valid_bare_path(path):
    if not path:
        return False
    for ch in path:
        if ch in _PATH_PUNCTUATION:
            continue
        category = unicodedata.category(ch)
        if ch == "�" or not (category[0] in "LM" or category == "Nd"):
            return False
    return True


def _split_coordinate(lexeme):
    match = _COORDINATE_TAIL.search(lexeme)
    path, start, end = lexeme, "", ""
    if match:
        path, start, end = lexeme[:match.start()], match.group(1), match.group(2) or ""
    return path, start, end, bool(path) and ":" not in path


def _is_malformed_coordinate(member):
    head = member.split(":", 1)[0]
    return bool(_MALFORMED_COORDINATE.match(member)) and head.strip("0123456789.") != ""


def _is_attempt(path, start, indexed, known):
    if indexed or path in (".", "./"):
        return False
    if start and ("." in path or "/" in path) and path.strip("0123456789.") != "":
        return True
    return known(path)


def _judge_member(member, indexed, known):
    """(attempt, plausible): implausible means no citation can contain this member."""
    if member.lower().startswith(_BARE_PREFIXES):
        return False, False
    if not indexed and _is_malformed_coordinate(member):
        return True, True
    if "#" in member:
        return False, bool(_HANDLE_SHAPED.match(member))
    path, start, _, ok = _split_coordinate(member)
    if not ok or not _valid_bare_path(path):
        return False, False
    return _is_attempt(path, start, indexed, known), True


def _set_lines(citation, start, end):
    if not start:
        return ""
    line = int(start)
    if line <= 0 or line > MAX_LINE:
        return "invalid_range"
    citation.line = line
    if not end:
        return ""
    last = int(end)
    if last < line or last > MAX_LINE:
        return "invalid_range"
    if last > line:
        citation.end_line = last
    return ""


def _terminal_kind(handle):
    kind = handle.split("#", 1)[0]
    return kind[kind.rfind(":") + 1:]


class _MemberReader:
    """Walks a citation body outside and across quoted strings."""

    def __init__(self, body, gate=None):
        self.body, self.at, self.gate = body, 0, gate

    def done(self):
        return self.at >= len(self.body)

    def skip_space(self):
        while not self.done() and self.body[self.at] in " \t":
            self.at += 1
        return self.at

    def lexeme(self):
        start = self.at
        while not self.done() and self.body[self.at] not in " \t,":
            self.at += 1
        return self.body[start:self.at]

    def at_quote(self):
        return not self.done() and self.body[self.at] == '"'

    def quoted(self):
        raw = ['"']
        self.at += 1
        while not self.done():
            ch = self.body[self.at]
            if ch == '"':
                self.at += 1
                try:
                    return json.loads("".join(raw) + '"'), True
                except ValueError:
                    return "", False
            if ch == "\\" and self.at + 1 < len(self.body):
                self.at += 1
                following = self.body[self.at]
                raw.append(following if following in "[]" else ch + following)
            else:
                raw.append(ch)
            self.at += 1
        return "", False

    def member(self):
        citation = Citation()
        if not self.at_quote():
            saved = self.at
            lexeme = self.lexeme()
            match = _HANDLE_WITH_COORDINATE.match(lexeme)
            if match:
                if self.gate is not None and not _known_kind(_terminal_kind(match.group(1))):
                    return None, ""
                citation.handle = match.group(1)
                code = _set_lines(citation, match.group(2) or "", match.group(3) or "")
                if code:
                    return None, code
                return self._after_handle(citation)
            if "#" in lexeme:
                if (self.gate is not None and not _is_malformed_coordinate(lexeme)
                        and not _known_kind(_terminal_kind(lexeme[:lexeme.index("#")]))):
                    return None, ""
                return None, "malformed_citation"
            self.at = saved
        attempt, code = self._path_ref(citation, self.gate)
        if code:
            return None, code
        result, code = self._excerpt(citation)
        return (result if attempt else None), code

    def _after_handle(self, citation):
        self.skip_space()
        if self.done() or self.body[self.at] == ",":
            return citation, ""
        if self.at_quote():
            saved = self.at
            text, ok = self.quoted()
            if not ok:
                return None, "malformed_citation"
            if self.done() or self.body[self.at] != ":":
                citation.excerpt = text
                return citation, ""
            self.at = saved
        _, code = self._path_ref(citation, None)
        if code:
            return None, code
        return self._excerpt(citation)

    def _path_ref(self, citation, gate):
        if self.at_quote():
            path, ok = self.quoted()
            if not ok or not path:
                return False, "malformed_citation"
            start, end = "", ""
            if not self.done() and self.body[self.at] == ":":
                tail = self.lexeme()
                match = _COORDINATE_TAIL.search(tail)
                if not match or match.group(0) != tail:
                    return False, "malformed_citation"
                start, end = match.group(1), match.group(2) or ""
            attempt = gate is None or (not gate[1] and (start != "" or gate[0](path)))
        else:
            path, start, end, ok = _split_coordinate(self.lexeme())
            if not ok or not _valid_bare_path(path):
                return False, "malformed_citation"
            attempt = gate is None or _is_attempt(path, start, gate[1], gate[0])
        if not attempt:
            return False, ""
        citation.path = path
        if start and citation.line:
            return False, "malformed_citation"
        return True, _set_lines(citation, start, end)

    def _excerpt(self, citation):
        self.skip_space()
        if not self.at_quote():
            return citation, ""
        text, ok = self.quoted()
        if not ok:
            return None, "malformed_citation"
        citation.excerpt = text
        return citation, ""


@dataclass
class _Node:
    """A closed node. prose means ordinary Markdown; issue means a damaged node."""

    kind: str = CITATION
    members: list = field(default_factory=list)  # (start, end, Occurrence) inside the node
    defects: list = field(default_factory=list)
    issue: str = ""
    prose: bool = False


def _parse_citation_group(body, attempt, indexed, known):
    node = _Node()
    reader = _MemberReader(body, gate=(known, indexed))
    while True:
        start = reader.skip_space()
        citation, code = reader.member()
        end = start + len(body[start:reader.at].rstrip(" \t\r"))
        if code and (attempt or code == "invalid_range"):
            return _Node(issue=code)
        if code:
            return _Node(prose=True)
        if citation is not None:
            node.members.append((start + 1, end + 1, Occurrence(CITATION, "", citation=citation)))
        else:
            node.defects.append((start + 1, end + 1))
        reader.skip_space()
        if reader.done():
            break
        if body[reader.at] != ",":
            return _Node(issue="malformed_citation") if attempt else _Node(prose=True)
        reader.at += 1
    return node if node.members else _Node(prose=True)


def _parse_verify(body):
    bad = _Node(kind=VERIFY, issue="invalid_verification")
    if not body.startswith("verify:"):
        return bad
    rest = body[len("verify:"):].lstrip(" \t")
    end = 0
    while end < len(rest) and "a" <= rest[end] <= "z":
        end += 1
    method, rest = rest[:end], rest[end:].lstrip(" \t")
    for separator in VERIFY_SEPARATORS:
        if rest.startswith(separator):
            rest = rest[len(separator):]
            break
    else:
        return bad
    # Go's TrimSpace set, which leaves the separators U+001C-U+001F in place.
    reason = rest.replace("\\[", "[").replace("\\]", "]").strip("\t\n\v\f\r \x85\xa0                　")
    if method not in VERIFY_METHODS or not reason or len(reason.encode("utf-8")) > MAX_REASON_BYTES:
        return bad
    return _Node(kind=VERIFY, members=[(0, 0, Occurrence(VERIFY, "", method=method, reason=reason))])


def _parse_artifact(body):
    bad = _Node(kind=ARTIFACT, issue="malformed_artifact")
    if not body.startswith("artifact:"):
        return bad
    node = _Node(kind=ARTIFACT)
    offset = 1 + len("artifact:")
    for raw in body[len("artifact:"):].split(","):
        identifier = raw.strip()
        if not _ARTIFACT_ID.match(identifier):
            return bad
        start = offset + raw.index(identifier)
        node.members.append((start, start + len(identifier), Occurrence(ARTIFACT, "", artifact=identifier)))
        offset += len(raw) + 1
    return node


def _parse_repair(body):
    bad = _Node(kind=REPAIR, issue="malformed_repair")
    if not body.startswith("cite:") or "=>" not in body:
        return bad
    target, value = body[len("cite:"):].split("=>", 1)
    target, value = target.strip(), value.strip()
    if not _REPAIR_TARGET.match(target) or not value:
        return bad
    occurrence = Occurrence(REPAIR, "", target=target)
    if value.startswith("<") and value.endswith(">"):
        occurrence.url = value[1:-1]
        if not _is_web(occurrence.url):
            return bad
    else:
        reader = _MemberReader(value)
        citation, code = reader.member()
        if code or not reader.done():
            return bad
        occurrence.citation = citation
    return _Node(kind=REPAIR, members=[(0, 0, occurrence)])


class _Scanner:
    """Recognizes one custom node from its '[', one character at a time."""

    def __init__(self, window, indexed, known):
        self.window, self.indexed, self.known = window, indexed, known
        self.buf = ["["]
        self.bytes = 1
        self.mode = _LEXEME
        self.confirmed = self.attempt = False
        self.member_start = 1
        self.after_member = self.after_quote = self.in_quote = False
        self.escaped = self.in_angle = False
        self.depth = self.skip_until = 0
        match = _NAMESPACED_HEAD.match(window)
        if match and _known_kind(match.group(1)):
            return
        match = _RESERVED_HEAD.match(window)
        if match:
            self.confirmed = True
            self.mode = _ARTIFACT if match.group(1) == "artifact" else _VERIFY
        elif _REPAIR_HEAD.match(window):
            self.confirmed = True
            self.mode = _REPAIR

    def kind(self):
        return _MODE_KIND.get(self.mode, CITATION)

    def text(self):
        return "".join(self.buf)

    def _push(self, ch):
        self.buf.append(ch)
        self.bytes += len(ch.encode("utf-8"))

    def feed(self, ch):
        if ch == "\n":
            return _UNTERMINATED if self.confirmed else _RELEASE
        if self.bytes + len(ch.encode("utf-8")) > MAX_NODE_BYTES:
            return _OVERFLOW if self.confirmed else _RELEASE
        return self._feed_lexeme(ch) if self.mode == _LEXEME else self._feed_confirmed(ch)

    def _member(self):
        return "".join(self.buf[self.member_start:])

    def _feed_lexeme(self, ch):
        if self.in_quote:
            self._push(ch)
            if self.escaped:
                self.escaped = False
            elif ch == "\\":
                self.escaped = True
            elif ch == '"':
                self.in_quote, self.after_quote = False, True
            return _PENDING
        if self.after_quote:
            return self._feed_after_quote(ch)
        if self.after_member:
            return self._feed_after_member(ch)
        if ch == '"':
            if len(self.buf) != self.member_start:
                return _RELEASE
            self.in_quote = True
        elif ch == "#":
            member = self._member()
            if _known_kind(member[member.rfind(":") + 1:]):
                return self._confirm(_CITATION, ch)
        elif ch in "], \t\r":
            return self._end_member(ch)
        elif not (ord(ch) >= 0x80 or ch.isascii() and ch.isalnum() or ch in _LEXEME_BYTES):
            return _RELEASE
        self._push(ch)
        return _PENDING

    def _feed_after_quote(self, ch):
        self.after_quote = False
        if ch == ":":
            following = len(self.buf) + 1
            if self.indexed or following >= len(self.window) or not "0" <= self.window[following] <= "9":
                return _RELEASE
            return self._confirm(_CITATION, ch)
        return self._feed_after_member(ch)

    def _feed_after_member(self, ch):
        if ch in " \t\r":
            self._push(ch)
            self.after_member = True
            return _PENDING
        if ch == '"':
            self.after_member = False
            if self.attempt:
                return self._confirm(_CITATION, ch)
            self.in_quote = True
            self._push(ch)
            return _PENDING
        if ch == ",":
            return self._next_member(ch)
        if ch == "]":
            self._push(ch)
            return _CLOSED
        return _RELEASE

    def _end_member(self, ch):
        member = self._member()
        if not member:
            if ch in " \t" and self.member_start > 1:
                self._push(ch)
                self.member_start = len(self.buf)
                return _PENDING
            return _RELEASE
        attempt, plausible = _judge_member(member, self.indexed, self.known)
        if not plausible:
            return _RELEASE
        self.attempt = self.attempt or attempt
        if ch == "]":
            self._push(ch)
            return _CLOSED
        if ch == ",":
            return self._next_member(ch)
        self._push(ch)
        self.after_member = True
        return _PENDING

    def _next_member(self, ch):
        if self.attempt:
            return self._confirm(_CITATION, ch)
        self.after_member = False
        self._push(ch)
        self.member_start = len(self.buf)
        return _PENDING

    def _confirm(self, mode, ch):
        self.mode = mode
        self.confirmed = True
        self.in_quote = ch == '"'
        self.after_member = self.after_quote = False
        self._push(ch)
        return _PENDING

    def _feed_confirmed(self, ch):
        self._push(ch)
        if len(self.buf) <= self.skip_until:
            return _PENDING
        if self.escaped:
            self.escaped = False
            return _PENDING
        if ch == "\\":
            self.escaped = True
            return _PENDING
        if self.mode == _VERIFY:
            # A reason is prose: its quotes mean nothing, brackets nest, code is opaque.
            if ch == "`":
                self._skip_code_span()
            elif ch == "[":
                self.depth += 1
            elif ch == "]":
                if self.depth == 0:
                    return _CLOSED
                self.depth -= 1
            return _PENDING
        if self.mode == _ARTIFACT:
            return _CLOSED if ch == "]" else _PENDING
        return self._feed_quote_aware(ch)

    def _skip_code_span(self):
        start = len(self.buf) - 1
        run = start
        while run < len(self.window) and self.window[run] == "`":
            run += 1
        length, at = run - start, run
        while at < len(self.window):
            if self.window[at] != "`":
                at += 1
                continue
            end = at
            while end < len(self.window) and self.window[end] == "`":
                end += 1
            if end - at == length:
                self.skip_until = end
                return
            at = end + 1
        self.skip_until = run

    def _feed_quote_aware(self, ch):
        if self.in_quote:
            self.in_quote = ch != '"'
        elif self.in_angle:
            self.in_angle = ch != ">"
        elif ch == '"':
            self.in_quote = True
        elif ch == "<" and self.mode == _REPAIR:
            self.in_angle = True
        elif ch == "]":
            return _CLOSED
        return _PENDING


def _url_run_end(text, at, end):
    while at < end:
        ch = text[at]
        if ch == "<" or ch in _UNICODE_SPACE:
            return at
        at += 1
    return end


def _backpedal_url(url):
    balanced = set()
    end = len(url)
    i = 0
    while i < end:
        if url[i] == "(":
            closer = url.find(")", i)
            if closer < 0:
                end = i
                break
            i = closer
            balanced.add(i)
        i += 1
    url = url[:end]
    while url:
        if url[-1] == ";":
            at = len(url) - 2
            while at >= 0 and url[at].isascii() and url[at].isalnum():
                at -= 1
            if at >= 0 and url[at] == "&" and at < len(url) - 2:
                url = url[:at]
                continue
        if url[-1] not in "?!.,:;*_'\"~)" or len(url) - 1 in balanced:
            return url
        url = url[:-1]
    return url


def _workspace_citation(destination):
    citation = Citation()
    path, _, fragment = destination.partition("#")
    path = path.partition("?")[0]
    match = _LINE_FRAGMENT.match(fragment)
    if match:
        code = _set_lines(citation, match.group(1), match.group(2) or "")
        if code:
            return None, code
    try:
        citation.path = urllib.parse.unquote(path, errors="strict")
    except UnicodeDecodeError:
        citation.path = path
    return citation, ""


def _source_citation(destination):
    parsed = urllib.parse.urlsplit(destination)
    relative = parsed.path.removeprefix("/")
    if not parsed.netloc or re.search(r"[@\s]", parsed.netloc) or not relative:
        return None, "malformed_citation"
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if len(query) > 1 or (len(query) == 1 and not query.get("job_id", [""])[0]):
        return None, "malformed_citation"
    return _workspace_citation(relative + ("#" + parsed.fragment if parsed.fragment else ""))


class _Builder:
    def __init__(self, known):
        self.document = Document()
        self.known = known
        self.first_verify = None
        self.dropped = 0

    def add_issue(self, code, kind, token):
        if len(self.document.issues) >= MAX_OCCURRENCES:
            self.dropped += 1
            return
        self.document.issues.append(Issue(code, kind, _bounded(token)))

    def closed(self, token, node):
        if len(self.document.occurrences) + len(node.members) > MAX_OCCURRENCES:
            self.dropped += len(node.members)
            return
        for start, end, occurrence in node.members:
            occurrence.token = token[start:end] if end > start else token
            if occurrence.kind == VERIFY:
                self._check_verification(occurrence)
            self.document.occurrences.append(occurrence)
        for start, end in node.defects:
            self.add_issue("malformed_citation", CITATION, token[start:end])

    def _check_verification(self, occurrence):
        if self.first_verify is None:
            self.first_verify = occurrence
        elif (self.first_verify.method, self.first_verify.reason) != (occurrence.method, occurrence.reason):
            self.add_issue("conflicting_verification", VERIFY, occurrence.token)

    def link(self, token, destination):
        occurrence = Occurrence(WEB_LINK, token, destination=destination)
        if not _is_web(destination):
            if destination[:7].lower() == "source:":
                citation, code = _source_citation(destination)
            elif not destination or destination.startswith("#") or _ANY_SCHEME.match(destination):
                return
            else:
                citation, code = _workspace_citation(destination)
            if code:
                self.add_issue(code, WORKSPACE_LINK, token)
                return
            occurrence.kind, occurrence.citation = WORKSPACE_LINK, citation
            if not citation.line and not self.known(citation.path):
                return
        if len(self.document.occurrences) >= MAX_OCCURRENCES:
            self.dropped += 1
            return
        self.document.occurrences.append(occurrence)


class _InlineScan:
    """Offers one inline block's text to the node scanner, passing literal zones."""

    def __init__(self, source, zones, builder):
        self.source, self.zones, self.builder = source, zones, builder
        self.pos = self.next = self.start = 0
        self.last_open = -1
        self.escaped = False
        self.scanner = None
        self.skip_line = False
        self.overflow = ("", CITATION)

    def run(self):
        while True:
            self._text()
            if not self._finish():
                return

    def _text(self):
        source, end = self.source, len(self.source)
        while self.pos < end:
            at = self.pos
            while self.next > 0 and self.zones[self.next - 1][1] > at:
                self.next -= 1
            while self.next < len(self.zones) and self.zones[self.next][1] <= at:
                self.next += 1
            idle = self.scanner is None and not self.skip_line
            if idle and self.next < len(self.zones) and self.zones[self.next][0] <= at:
                self._enter_zone(at)
                continue
            if idle and self._enter_bare_link(at):
                continue
            ch = source[at]
            if idle:
                if ch == "[":
                    self.last_open = at
                elif ch == "]":
                    self.last_open = -1
            self._feed(ch)

    def _enter_zone(self, at):
        start, end, destination = self.zones[self.next]
        # A node that ran into this zone took its start; the rest stays literal.
        if start >= at and destination is not None:
            self.builder.link(self.source[start:end], destination)
        self.pos, self.escaped = end, False

    def _enter_bare_link(self, at):
        source = self.source
        if source[at] not in "hHfFwW":
            return False
        for prefix in _BARE_PREFIXES:
            head = source[at:at + len(prefix)]
            following = source[at + len(prefix):at + len(prefix) + 1]
            if head.lower() == prefix and following and (following == "-" or following.isascii() and following.isalnum()):
                break
        else:
            return False
        run_end = _url_run_end(source, at, len(source))
        if len(source[at:run_end].encode("utf-8")) > MAX_NODE_BYTES:
            self.builder.add_issue("node_too_long", WEB_LINK, source[at:run_end])
            self.pos, self.escaped = run_end, False
            return True
        url = _backpedal_url(source[at:run_end])
        if not url:
            return False
        if url.endswith("]") and 0 <= self.last_open < at and "\n" not in source[self.last_open:at]:
            url = url[:-1]
            self.last_open = -1
        destination = "http://" + url if url[:4].lower() == "www." else url
        if _is_web(destination):
            self.builder.link(url, destination)
        self.pos, self.escaped = at + len(url), False
        return True

    def _feed(self, ch):
        if self.skip_line:
            if ch == "\n":
                self._finish_skip()
                self.skip_line = False
                return
            self.pos += 1
        elif self.scanner is not None:
            self._feed_node(ch)
        elif ch == "[" and not self.escaped:
            self.start = self.pos
            self.scanner = _Scanner(self._window(), self._preceded_by_identifier(), self.builder.known)
            self.pos += 1
        else:
            self.escaped = not self.escaped and ch == "\\"
            self.pos += 1

    def _window(self):
        newline = self.source.find("\n", self.pos)
        limit = len(self.source) if newline < 0 else newline + 1
        return self.source[self.pos:min(limit, self.pos + MAX_NODE_BYTES)]

    def _preceded_by_identifier(self):
        if self.pos == 0:
            return False
        ch = self.source[self.pos - 1]
        return ch == "_" or (ch.isascii() and ch.isalnum())

    def _feed_node(self, ch):
        result = self.scanner.feed(ch)
        if result == _CLOSED:
            self.pos += 1
            self._close_node()
        elif result == _RELEASE:
            self._release()
        elif result == _UNTERMINATED:
            self._damage("unterminated_node")
            self.scanner = None
        elif result == _OVERFLOW:
            self.overflow = (_bounded(self.scanner.text()), self.scanner.kind())
            self.scanner = None
            self.skip_line = True
            self.pos += 1
        else:
            self.pos += 1

    def _release(self):
        self.scanner = None
        self.escaped = False
        self.pos = self.start + 1

    def _close_node(self):
        scanner = self.scanner
        token = scanner.text()
        body = token[1:-1]
        if scanner.mode == _VERIFY:
            node = _parse_verify(body)
        elif scanner.mode == _ARTIFACT:
            node = _parse_artifact(body)
        elif scanner.mode == _REPAIR:
            node = _parse_repair(body)
        else:
            node = _parse_citation_group(body, scanner.confirmed or scanner.attempt, scanner.indexed, self.builder.known)
        if node.prose:
            self._release()
            return
        self.scanner = None
        self.escaped = False
        if node.issue:
            self.builder.add_issue(node.issue, node.kind, token)
        else:
            self.builder.closed(token, node)

    def _damage(self, code):
        end = self.pos
        if end > self.start and self.source[end - 1] == "\r":
            end -= 1
        self.builder.add_issue(code, self.scanner.kind(), self.source[self.start:end])

    def _finish_skip(self):
        self.builder.add_issue("node_too_long", self.overflow[1], self.overflow[0])

    def _finish(self):
        """End of input, which matches a newline; True means a released bracket needs rescanning."""
        if self.skip_line:
            self._finish_skip()
            self.skip_line = False
        elif self.scanner is not None:
            if not self.scanner.confirmed:
                self._release()
                return True
            self._damage("unterminated_node")
            self.scanner = None
        return False


def _record_zones(rule, kinds):
    """Wrap an inline rule to note where its node sits; the parser keeps no such fact."""

    def recording(state, silent):
        scan = state.env.get("scan")
        before, tokens = state.pos, len(state.tokens)
        matched = rule(state, silent)
        if matched and not silent and scan is not None and state.src is scan["source"]:
            pushed = state.tokens[tokens:]
            if any(t.type in kinds for t in pushed):
                destination = None
                link = next((t for t in pushed if t.type == "link_open"), None)
                if link is not None:
                    destination = link.attrGet("href")
                    if link.markup == "autolink" and destination != state.src[before + 1:state.pos - 1]:
                        destination = None
                scan["zones"].append((before, state.pos, destination))
        return matched

    return recording


def _build_parser():
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    md.disable("emphasis")
    md.core.ruler.enableOnly(["normalize", "block"])
    # Host semantics: destinations stay as written and every scheme parses as a link.
    md.normalizeLink = md.normalizeLinkText = lambda url: url
    md.validateLink = lambda url: True
    wrapped = {"backticks": {"code_inline"}, "link": {"link_open"}, "image": {"image"},
               "autolink": {"link_open"}, "html_inline": {"html_inline"}}
    for rule in md.inline.ruler.__rules__:
        if rule.name in wrapped:
            rule.fn = _record_zones(rule.fn, wrapped[rule.name])
    return md


_MD = _build_parser()


def _report_fence(tokens, source):
    """The body of a closed top-level json fence that ends the document."""
    last = tokens[-1] if tokens else None
    if last is None or last.type != "fence" or last.level != 0 or last.info.strip().lower() != "json":
        return None
    lines = source.split("\n")
    closer = lines[last.map[1] - 1].strip() if last.map[1] - 1 > last.map[0] else ""
    closed = closer and set(closer) == {last.markup[0]} and len(closer) >= len(last.markup)
    return last.content if closed else None


def _wrapper_body(tokens, source):
    """The body of a whole-response Markdown fence, whose span is unindented."""
    if len(tokens) != 1 or tokens[0].type != "fence":
        return None
    fence = tokens[0]
    opener = source.split("\n")[fence.map[0]]
    if fence.level != 0 or fence.info.strip().lower() not in ("markdown", "md") or not fence.content or opener[:1] in (" ", "\t"):
        return None
    return fence.content


def parse(source, known_path=None):
    """Read one Markdown text into occurrences and issues, in document order."""
    known = known_path or (lambda name: False)
    builder = _Builder(known)
    document = builder.document
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        document.issues.append(Issue("source_budget_exceeded", "", _bounded(source)))
        return document
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    tokens = _MD.parse(source)
    body = _wrapper_body(tokens, source)
    if body is not None:
        source = body
        tokens = _MD.parse(source)
    document.report_fence = _report_fence(tokens, source)
    references = _references(source)
    for token in tokens:
        if token.type != "inline":
            continue
        scan = {"source": token.content, "zones": []}
        _MD.inline.parse(token.content, _MD, {"scan": scan, "references": references}, [])
        _InlineScan(token.content, sorted(scan["zones"], key=lambda z: z[0]), builder).run()
    if builder.dropped:
        document.issues.append(Issue("occurrence_budget_exceeded", "", str(builder.dropped)))
    return document


def _references(source):
    env = {}
    _MD.parse(source, env)
    return env.get("references", {})
