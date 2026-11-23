"""실시간 애매함 분석(meetalign.live)의 규칙 안전망 — 모델과 상관없이 코드가 직접 짚는 두 가지.

1. 일정 충돌(find_conflicts): "앞에서 안 된다고 한 요일·시간"과 겹치는 일정 제안.
2. 미룸(find_deferrals): "장소는 이따 단톡에서 정해요"처럼 정하지 않고 나중으로 넘긴 말.

AI 모델이 가벼우면(무료 한도에 걸려 flash-lite로 넘어갔을 때) "저는 목요일 7시에 약속이 있어서 못할 거 같아요" 뒤의
"그럼 목요일 7시에 다음 회의 잡을까요?"를 그대로 넘겼다 — 사정은 뽑아 놓고도 새 제안과 맞대 보지 않았다. 이런 충돌은
요일과 시간만 맞대 보면 알 수 있어서, 모델과 상관없이 코드가 직접 짚는다.

실시간 조각은 말 한마디 단위라 한 문장이 여러 조각에 걸친다("저는 목요일 7시에" / "약속이 있어서" / "못할 거
같아요"). 그래서 요일이 나온 말에서 시작해 뒤의 말 두 개까지를 한 덩어리로 보되, 다음 요일이 나오면 거기서 끊는다.
"""

import re

WEEKDAY = re.compile(r"([월화수목금토일])요일")
_TIME = r"(?:(오전|오후|아침|저녁|밤)\s*)?(\d{1,2})\s*시(?:\s*(반|\d{1,2}\s*분))?"
# 요일 바로 뒤(조사 등 몇 글자 안)에 오는 시간 또는 시간 범위. "목요일에는 3시부터 4시 반까지", "목요일 오후 4시".
SLOT_AFTER_WEEKDAY = re.compile(
    r"^[^\d월화수목금토일.?!]{0,12}?" + _TIME + r"(?:\s*(?:부터|~|-|에서)\s*" + _TIME + r")?"
)
# 안 된다는 말. "안 돼요", "못할 거 같아요", "약속이 있어서", "수업이 있어서", "알바가 있어서" 등.
UNAVAILABLE = re.compile(
    r"안\s*[돼되될]|못\s*(?:해|하|할|가|갈|와|올|만나|모이|모여|있|나가)|못할|"
    r"(?:약속|수업|강의|알바|아르바이트|일정|선약|일|시험|과외|근무)(?:이|가)?\s*있|선약|바빠|바쁘|어려울|어려워|불가"
)
# 일정을 제안하거나 정하는 말.
PROPOSAL = re.compile(r"회의|모이|모여|모임|만나|만날|잡|어때|어떠|할게요|하죠|합시다|하자|로\s*해요|로\s*하|정할게|확정")
# 제안 덩어리에 이런 말이 있으면 그 요일을 피하자는 말이다("목요일 말고 금요일").
AVOIDING = re.compile(r"말고|빼고|아니고|피해|제외")
FOLLOWING_LINES = 2
# 규칙이 올린 애매한 점의 사유 머리말. AI가 같은 충돌을 다른 표현으로 다시 올리면 이것으로 알아보고 뺀다(meetalign.live).
REASON_PREFIX = "앞에서 안 된다고 한 시간("


def _hour(marker, hour, extra):
    value = int(hour)
    if marker in ("오후", "저녁", "밤") and value < 12:
        value += 12
    if extra == "반":
        return value + 0.5, bool(marker)
    if extra:
        return value + int(re.sub(r"\D", "", extra)) / 60, bool(marker)
    return value, bool(marker)


def _slot(text):
    """요일 바로 뒤 텍스트에서 (시작, 끝, 오전·오후 표시가 있었나, 맞은 글자 수). 시간이 없으면 None."""
    match = SLOT_AFTER_WEEKDAY.match(text)
    if not match:
        return None
    start, marked = _hour(*match.group(1, 2, 3))
    if match.group(5):
        end, end_marked = _hour(match.group(4) or match.group(1), match.group(5), match.group(6))
        marked = marked or end_marked
        if end <= start:
            end += 12  # "3시부터 4시 반" 같은 말은 늘 오후라 생략한 경우가 많다 — 끝이 앞서면 12시간 뒤로 본다
    else:
        end = start + 1  # "7시에 약속" — 정확한 끝을 모르니 한 시간으로 본다
    return start, end, marked, match.end()


def _overlaps(constraint, proposal):
    """두 시간대가 겹치나. 한쪽이라도 오전·오후를 말하지 않았으면 12시간 단위로 접어서 본다(4시 = 오후 4시)."""
    (c_start, c_end, c_marked), (p_start, p_end, p_marked) = constraint, proposal
    if not (c_marked and p_marked):
        shift = (p_start // 12 - c_start // 12) * 12
        c_start, c_end = c_start + shift, c_end + shift
    return c_start < p_end and p_start < c_end


def _segments(lines):
    """[(줄 번호, 요일, 요일부터 다음 요일 앞까지의 덩어리)] — 덩어리는 그 줄과 뒤의 말 FOLLOWING_LINES개를 잇는다."""
    for index, line in enumerate(lines):
        joined = " ".join(lines[index:index + 1 + FOLLOWING_LINES])
        for match in WEEKDAY.finditer(line):
            rest = joined[match.start():]
            following = WEEKDAY.search(rest, match.end() - match.start())
            yield index, match.group(1), rest[:following.start()] if following else rest


def is_rule_reason(reason):
    """규칙이 올린 애매한 점인가(사유 머리말로 알아본다)."""
    return reason.startswith(REASON_PREFIX) or DEFERRAL_PREFIX in reason


def same_conflict(left, right):
    """두 표현이 같은 요일·같은 시각을 가리키나("다음 주 목요일 4시"와 "목요일 오후 4시")."""
    days = [set(WEEKDAY.findall(text)) for text in (left, right)]
    hours = [set(re.findall(r"(\d{1,2})\s*시", text)) for text in (left, right)]
    return bool(days[0] & days[1]) and bool(hours[0] & hours[1])


def _short(text, limit=24):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def find_conflicts(lines, first_new, last_new=None):
    """lines(시간순 말 목록)에서 first_new~last_new 줄에 걸친 일정 제안 중, 그보다 앞에서 누군가 안 된다고 한 요일·시간과
    겹치는 것 → [(phrase, reason, topic)]. phrase는 제안에 나온 "요일 시간" 그대로다(녹취 인용 검사를 통과하게).

    요일이 지난번에 판단한 줄에 있어도 덩어리가 새 줄에 걸치면 다시 본다 — "그럼 목요일 7시에"를 본 뒤에야 "다음 회의
    잡을까요?"가 들어오는 일이 잦다. 같은 것을 두 번 올리지 않는 일은 부르는 쪽이 한다."""
    last_new = len(lines) - 1 if last_new is None else last_new
    constraints, found, seen = [], [], set()
    for index, weekday, segment in _segments(lines):
        slot = _slot(segment[3:])
        if UNAVAILABLE.search(segment):
            constraints.append((index, weekday, slot, segment))
            continue
        if not (first_new - FOLLOWING_LINES <= index <= last_new) or not PROPOSAL.search(segment) or AVOIDING.search(segment):
            continue
        for c_index, c_weekday, c_slot, c_segment in constraints:
            if c_index >= index or c_weekday != weekday:
                continue
            if c_slot and not (slot and _overlaps(c_slot[:3], slot[:3])):
                continue
            phrase = segment[:3 + slot[3]].strip() if slot else segment[:3]
            key = (phrase, c_index)
            if key in seen:
                continue
            seen.add(key)
            told = _short(c_segment[:3 + c_slot[3]] if c_slot else c_segment[:3])
            found.append((phrase, REASON_PREFIX + "%s)과 겹쳐요. 다시 정해 주세요" % told, "일정 충돌 · %s요일" % weekday))
            break
    return found


# 미룸: 언제(이따·나중에)나 어디서(단톡에서) 정하겠다고 넘긴 말. "정해요/정할게요/얘기해요"가 바로 뒤에 와야 한다 —
# "나중에 봬요" 같은 인사, "단톡에 올릴게요" 같은 할 일은 잡지 않는다.
DEFERRAL = re.compile(
    r"(?:이따가?|나중에|다음에|다음 시간에|따로|단톡방?에서|카톡(?:방)?에서|채팅방에서|톡으로)"
    r"[^.?!]{0,8}?(?:정해요|정하죠|정해 봐요|정해봐요|정할게요|정하기로|정합시다|정하자|얘기해요|이야기해요|얘기하죠)"
)
# 미룸 사유의 머리말 — 같은 미룸을 다른 표현으로 다시 올리지 않게 알아본다(meetalign.live).
DEFERRAL_PREFIX = "정하지 않고 미뤘어요"


def find_deferrals(lines, first_new, last_new=None):
    """first_new~last_new 줄에서 결정을 미룬 말 → [(phrase, reason, topic)]. 무엇을 미뤘는지는 앞의 말("장소는")로 보여 준다."""
    last_new = len(lines) - 1 if last_new is None else last_new
    found = []
    for line in lines[first_new:last_new + 1]:
        for match in DEFERRAL.finditer(line):
            # 무엇을 미뤘는지 — "장소는 이따…"의 '장소'. 없으면 앞의 말 몇 글자로 보여 준다(사유가 서로 달라야 다른
            # 미룸이 같은 것으로 걸러지지 않는다).
            topic = re.search(r"([가-힣A-Za-z]{1,8})(?:은|는)\s*$", line[:match.start()])
            what = topic.group(1) if topic else _short(line[:match.start()], 12) or _short(line, 12)
            found.append((match.group(0), "%s — %s. 누가 언제까지 정할지 정해 주세요" % (what, DEFERRAL_PREFIX),
                          "%s 미룸" % what if topic else "미룬 안건"))
    return found
