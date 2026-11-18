"""AI가 쓴 추천 문장을 그대로 진행자에게 내놓을 만한지 보는 검사, 그리고 AI가 채운 값의
출처 검증·화면 표시용 순수 함수.

프롬프트에서 "원문에 없는 값을 지어내지 마라"는 제한을 풀었기 때문에 필요해진 모듈이다.
채우는 것을 허용하면 AI는 두 방향으로 미끄러진다 — 아무것도 못 채우고 근거를 그대로
옮겨 쓰거나, 서로 다른 안건에 같은 문장을 붙인다. 그 셋을 여기서 규칙으로 잡는다.

이 파일은 Django와 LLM을 모른다(import는 re 하나뿐이고, 테스트가 그걸 지킨다).
검사가 모델·네트워크 없이 도니까 같은 입력이면 늘 같은 결과가 나오고, 재생성 흐름 없이
검사만 떼어 시험해 볼 수 있다. 추출 모듈에 이미 같은 뜻의 정규식이 있지만 옮겨 오지 않고
여기 다시 적었다 — 같은 시기에 다른 작업자가 그 파일의 호출 흐름을 고치고 있어서, 두 작업이
한 줄에서 부딪히지 않게 하는 쪽을 택했다(두 변경이 모두 머지된 뒤에 하나로 정리한다).
"""

import re


# ---------------------------------------------------------------------------
# 비교용 문자열과 토큰
# ---------------------------------------------------------------------------


def squash(text):
    """공백을 모두 지운 비교용 문자열. 전사본과 AI 문장은 띄어쓰기가 들쭉날쭉해서다."""
    return re.sub(r"\s+", "", text or "")


# 화자 분리 표시("화자 A:"). 화자 분리가 한 사람의 말을 중간에서 끊어 다음 줄 다른 화자에게 붙이는 일이 잦다
# ("…그냥 그걸로 ⏎ 화자 D: 하겠습니다"). AI는 근거를 옮길 때 끊긴 말을 이어 붙이므로, 원문 대조는 이 표시를 지우고 한다.
SPEAKER_LABEL = re.compile(r"화자\s*[0-9A-Za-z?]{1,3}\s*:")
# 문장 속 화자 기호("화자 D가 …") — 콜론 없이 담당자로 쓰인 것.
SPEAKER_NAME = re.compile(r"화자\s*[0-9A-Za-z]{1,3}(?![0-9A-Za-z:])")
# AI가 떨어진 발언 여러 개를 " / "로 묶어 근거 하나로 보낼 때의 구분자.
QUOTE_SEPARATOR = re.compile(r"\s+/\s+")


def squash_transcript(text):
    """원문 대조용 문자열 — 화자 표시와 공백을 모두 지운다."""
    return squash(SPEAKER_LABEL.sub(" ", text or ""))


def quote_in_source(quote, squashed_source):
    """AI가 인용한 원문 구절이 전사본에 있는가. squashed_source는 squash_transcript(전사본)이다.

    " / "로 묶인 인용은 조각마다 따로 찾고, 모든 조각이 있어야 있다고 본다. 화자 표시와 띄어쓰기 차이는 무시하되
    글자는 그대로여야 한다 — 원문에 없는 말을 지어낸 인용은 여전히 걸러진다.
    """
    parts = [squash_transcript(part) for part in QUOTE_SEPARATOR.split(quote or "")]
    parts = [part for part in parts if part]
    return bool(parts) and all(part in squashed_source for part in parts)


def bigrams(text):
    """공백을 지운 뒤 두 글자씩 끊어 만든 집합. 한국어는 토큰 경계가 흐려서 글자 단위로 센다."""
    squashed = squash(text)
    return {squashed[i:i + 2] for i in range(len(squashed) - 1)}


# 날짜·시각·수량은 추출 모듈의 규칙과 같은 뜻이다(조사를 봐서 기한과 기간을 가린다).
DATE_EXPR = re.compile(
    r"\d{1,2}\s*월\s*\d{1,2}\s*일"
    r"|\d{1,2}\s*월(?=\s*(?:까지|에|부터|,|$))"
    r"|\d{1,2}\s*일(?=\s*(?:까지|에|부터|,|$))"
    r"|[월화수목금토일]요일"
    r"|\d+\s*주\s*(?:뒤|후|내)?"
)
TIME_EXPR = re.compile(r"\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?")
QUANTITY_EXPR = re.compile(r"\d+\s*(?:개(?!월)|명|건|퍼센트|%)")
# 맨 숫자. 위의 셋은 "경고를 띄울 만한 표현"을 고르느라 일부러 좁게 잡는다 — "30일로"처럼
# 조사가 붙은 기간은 날짜로 세지 않는다. 하지만 "한 달 정도"를 "30일로"로 바꾼 것은 분명히
# AI가 채운 값이므로, 새 정보를 셀 때는 맨 숫자까지 본다.
NUMBER_EXPR = re.compile(r"\d+")


def _tokens(expr, text):
    return {squash(match.group(0)) for match in expr.finditer(text or "")}


def date_tokens(text):
    return _tokens(DATE_EXPR, text)


def time_tokens(text):
    return _tokens(TIME_EXPR, text)


def quantity_tokens(text):
    return _tokens(QUANTITY_EXPR, text)


def number_tokens(text):
    return _tokens(NUMBER_EXPR, text)


def overlap_ratio(part, whole):
    """part가 whole에 얼마나 담겨 있나 — part 기준이라 길이 차이에 휘둘리지 않는다."""
    if not part:
        return 0.0
    return len(part & whole) / len(part)


def jaccard(left, right):
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# ---------------------------------------------------------------------------
# ① AI가 채운 값의 출처
# ---------------------------------------------------------------------------

# AI가 주장할 수 있는 출처. 서버가 대조해서 증명된 것만 남기고 나머지는 전부 SOURCE_GUESS가 된다.
SOURCE_QUOTE = "원문"
SOURCE_INFO = "회의정보"
SOURCE_GUESS = "AI추정"
FILL_SOURCES = (SOURCE_QUOTE, SOURCE_INFO, SOURCE_GUESS)


def verify_fill_sources(fills, source=None, info_text=""):
    """AI가 보낸 fills를 네 칸짜리 dict 목록으로 정리하고, 증명되지 않은 출처는 '추정'으로 내린다.

    AI는 "이건 원문에 있었다"면서 전사본에 없는 구절을 인용하기도 한다. 그 주장을 그대로 믿으면
    진행자 화면이 "원문 근거 있음"이라고 거짓말을 하게 되므로, 서버가 대조한 것만 인정한다.
    - "원문": quote를 전사본에서 찾을 수 있어야 하고, 채운 값이 그 quote 안에 들어 있어야 한다. 전사본에 있는
      아무 구절이나 붙여 놓고 짐작한 값을 "원문"이라고 하는 일이 있었다(담당 'demo_account1' — 원문, quote는
      그 이름이 없는 다른 문장).
    - "회의정보": 그 값이 실제 회의 정보(참석자·날짜·마감일·이월 안건) 안에 있어야 한다.
    - 그 밖의 값(모르는 라벨 포함): 추정.

    source가 None이면 대조할 원문이 없다는 뜻이라(가짜 응답 모드) 형식만 맞추고 출처는 그대로 둔다 —
    근거 검증을 건너뛰는 기존 파서 규칙과 같은 기준이다.

    형식이 틀린 원소는 조용히 버린다. 항목 하나의 형식 때문에 회의 전체를 실패시키지 않는다.
    """
    if not isinstance(fills, list):
        return []
    trust_claim = source is None
    in_source, in_info = squash_transcript(source), squash(info_text)
    verified = []
    for row in fills:
        if not isinstance(row, dict):
            continue
        value = str(row.get("value", "")).strip()
        if not value:
            continue
        claimed = str(row.get("source", "")).strip()
        quote = str(row.get("quote", "")).strip()
        if trust_claim:
            proven = claimed in FILL_SOURCES
        elif claimed == SOURCE_QUOTE:
            # 값이 화자 기호("화자 D")면 quote의 발화자 표시에 있어도 된다 — 그래서 표시를 지우지 않은 쪽도 본다.
            proven = quote_in_source(quote, in_source) and (
                squash_transcript(value) in squash_transcript(quote) or squash(value) in squash(quote))
        elif claimed == SOURCE_INFO:
            proven = bool(in_info) and squash(value) in in_info
        else:
            proven = False
        verified.append({
            "gap": str(row.get("gap", "")).strip(),
            "value": value,
            "source": claimed if proven else SOURCE_GUESS,
            "quote": quote,
        })
    return verified


# ---------------------------------------------------------------------------
# ② 추천 문장 검사
# ---------------------------------------------------------------------------

# 근거를 거의 그대로 옮겼다고 볼 선. 추천 문장은 근거를 바탕으로 쓰므로 어느 정도 겹치는 것이
# 정상이다. 0.8은 "문장 구조가 근거와 사실상 같다"에 해당한다.
COPY_RATIO = 0.8
# 두 결정의 문장이 같은 문장이라고 볼 선. 양쪽 길이를 함께 보는 자카드라 한쪽이 길어도 안전하다.
DUPLICATE_RATIO = 0.8

REASON_COPIED = "추천 문장이 원문 근거와 거의 같아요"
REASON_GUESSED_PERSON = "원문에 나오지 않은 사람을 담당자로 적었어요: %s"
REASON_NO_NEW_INFO = "원문에 없던 담당자·기한·수치가 하나도 없어요"
REASON_DUPLICATE = "다른 결정의 문장과 거의 같아요: %s"


def fill_segments(text, values):
    """문장을 [{"text", "estimated"}, ...]로 쪼갠다 — values에 든 값이 나오는 자리만 estimated다.

    진행자 화면이 그 자리만 다른 색으로 칠한다. 템플릿이 조각마다 {{ }}로 출력하므로 값에
    HTML이 섞여 있어도 그대로 문자로 보인다(여기서 마크업을 만들지 않는 이유다).

    긴 값을 먼저 찾는다 — "12월 5일"과 "5일"이 함께 오면 짧은 쪽이 먼저 걸려 긴 값이 쪼개진다.
    문장에 없는 값은 조용히 지나간다(AI가 문장과 다른 표기로 보낼 수 있다).
    """
    wanted = sorted({value for value in values or () if value}, key=len, reverse=True)
    if not wanted or not text:
        return [{"text": text, "estimated": False}]
    pattern = re.compile("|".join(re.escape(value) for value in wanted))
    segments, at = [], 0
    for match in pattern.finditer(text):
        if match.start() > at:
            segments.append({"text": text[at:match.start()], "estimated": False})
        segments.append({"text": match.group(0), "estimated": True})
        at = match.end()
    if at < len(text):
        segments.append({"text": text[at:], "estimated": False})
    return segments or [{"text": text, "estimated": False}]


def guessed_fills(fills):
    """AI추정으로 판정된 것만 — 화면에서 칠할 값과 경고에 쓸 빈칸 이름을 고르는 기준이다."""
    return [fill for fill in fills or () if fill.get("source") == SOURCE_GUESS and fill.get("value")]


def _names_in(text, participant_names):
    squashed_text = squash(text)
    return [name for name in participant_names or () if squash(name) and squash(name) in squashed_text]


def new_information(candidate, evidence, participant_names=(), source=None):
    """추천 문장에는 있고 근거에는 없는 값 토큰 — AI가 실제로 채운 것들.

    담당자 이름을 참석자 목록으로 보는 이유: 숫자가 전혀 없는 결정("민수가 발표 자료를 맡는다")도
    담당을 정했으면 빈칸을 채운 것이다. 다만 녹취 원문(source)에 나온 이름만 센다 — 명단에서 짐작한
    이름까지 '채웠다'고 인정하면 AI가 아무 이름이나 넣게 된다(실제로 거의 모든 추천 문장의 담당자가 맨
    앞의 진행자 계정이었다). source를 모르면(None) 예전처럼 명단만 본다.
    이름을 알 수 없어 녹취의 화자 기호("화자 D가 …")로 담당을 적은 것도 담당을 채운 것이므로 센다 — 근거
    줄 머리에 붙은 같은 기호는 발화자 표시일 뿐 담당을 정한 것이 아니어서 근거와 비교하지 않는다.
    """
    found = set()
    for extract in (date_tokens, time_tokens, quantity_tokens, number_tokens):
        found |= extract(candidate) - extract(evidence)
    in_evidence = squash(evidence)
    in_source = squash_transcript(source) if source is not None else None
    for name in _names_in(candidate, participant_names):
        squashed = squash(name)
        if squashed not in in_evidence and (in_source is None or squashed in in_source):
            found.add(squashed)
    found |= {squash(label) for label in SPEAKER_NAME.findall(candidate)}
    return found


def check_candidate(candidate, evidence, others=(), participant_names=(), source=None):
    """추천 문장을 쓸 만한가 → (통과 여부, 걸린 이유 목록).

    사유는 진행자 화면에 나가지 않는다 — 로그와 재생성 프롬프트에만 쓴다. 진행자에게는
    고칠 수 없는 내부 사정을 보여주는 대신, 통과한 문장이나 '직접 쓰기' 칸만 보여준다.

    후보가 없는 항목은 검사할 문장이 없으므로 통과로 본다 — 그 항목은 이미 직접 쓰기 흐름에 있다.
    """
    if not (candidate or "").strip():
        return True, []
    reasons = []
    candidate_grams = bigrams(candidate)
    if candidate_grams and overlap_ratio(candidate_grams, bigrams(evidence)) >= COPY_RATIO:
        reasons.append(REASON_COPIED)
    if not new_information(candidate, evidence, participant_names, source):
        reasons.append(REASON_NO_NEW_INFO)
    if source is not None:
        # 참석자 명단에는 있지만 녹취 어디에도 나오지 않는 이름 — 화자 기호만으로는 누구인지 알 수 없으니 짐작이다.
        in_source = squash_transcript(source)
        guessed = [name for name in _names_in(candidate, participant_names) if squash(name) not in in_source]
        if guessed:
            reasons.append(REASON_GUESSED_PERSON % ", ".join(guessed))
    for other in others or ():
        if other and jaccard(candidate_grams, bigrams(other)) >= DUPLICATE_RATIO:
            # 처음 걸린 문장만 알려 준다 — 재생성 프롬프트에는 어차피 다른 결정 문장 전체를 함께 넣는다.
            reasons.append(REASON_DUPLICATE % other)
            break
    return not reasons, reasons
