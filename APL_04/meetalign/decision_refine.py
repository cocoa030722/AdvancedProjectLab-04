"""AI 추천 문장을 진행자에게 내보내기 전에 되짚는 단계 — 검증 → 1회 재생성 → 폐기.

프롬프트에서 "원문에 없는 값을 지어내지 마라"는 제한을 풀면 빈칸이 채워지는 대신 두 가지가
새로 생긴다.
- 아무것도 못 채우고 근거를 거의 그대로 옮겨 쓴 문장. 진행자가 고를 것이 없다.
- 서로 다른 안건에 사실상 같은 문장. 공개되면 팀원이 중복된 결정을 받는다.
둘 다 모델에게 "그러지 마라"고 더 적는 것으로는 잘 막히지 않아서, 서버가 규칙으로 되짚고
걸린 것만 한 번 더 쓰게 한다(decision_checks가 판정, 이 모듈이 다시 쓰기).

왜 한 번만 다시 쓰나: 두 번째 시도도 걸리는 항목은 대개 원문에 채울 근거가 정말 없는
안건이다. 호출을 늘려도 같은 문장이 돌아오고 추출 시간만 길어지므로, 그 항목은 추천 문장을
버리고 '직접 쓰기' 흐름으로 보낸다(draft_hint). 진행자는 어차피 직접 쓰거나 다음 회의로
넘길 수 있다.

재생성 호출 자체가 실패했을 때(연결 끊김·읽을 수 없는 응답)는 원래 문장을 그대로 남긴다 —
검증에 걸렸다는 이유만으로 AI 장애 때 진행자 화면을 비우면 손해가 더 크다. 그 경우는
'폐기'와 구분해서 기록한다(check["kept"]).

호출 흐름은 decision_compiler.compile_specs가 저장 직전에 이 모듈을 한 줄로 부르는 것뿐이다.
"""

import json
import logging

from . import decision_checks, decision_compiler, services

logger = logging.getLogger(__name__)

# 다시 쓴 문장에도 남으면 안 되는 표현. 첫 추출에서 후보를 버리는 기준(_parse_candidates)과 같다 —
# 모호어가 남은 문장은 그대로 공개될 수 없으니 고를 의미가 없다.
VAGUE_PHRASES = decision_compiler.AMBIGUOUS_PHRASES + decision_compiler.REFERENT_PHRASES
REASON_STILL_VAGUE = "다시 쓴 문장에도 모호한 표현이 남았어요: '%s'"


def _checkable(item):
    """검사 대상인가 — 애매 항목의 추천 문장만 본다.

    바로 공개되는 명확 항목의 확정 문장은 원문 기준(parse_items의 근거 대조 + safety_warnings)을
    그대로 쓴다. 채우기를 허용한 것은 진행자가 고르는 추천 문장뿐이다.
    """
    return bool(item["ambiguous"] and item["candidates"])


def _other_sentences(items, index):
    """같은 회의의 다른 결정 문장들 — 중복 검사와 재생성 프롬프트에 쓴다."""
    sentences = []
    for other_index, item in enumerate(items):
        if other_index == index:
            continue
        sentences.extend(candidate["text"] for candidate in item["candidates"] if candidate["text"])
        if item["final"]:
            sentences.append(item["final"])
    return sentences


def _check(item, items, index, participants):
    return decision_checks.check_candidate(
        item["candidates"][0]["text"], item["evidence"],
        _other_sentences(items, index), participants,
    )


def _vague_reason(text):
    for phrase in VAGUE_PHRASES:
        if phrase in text:
            return REASON_STILL_VAGUE % phrase
    return ""


def _refine_prompt(items, flagged, source, meeting_info):
    """걸린 항목만 모아 다시 쓰게 하는 프롬프트.

    항목마다 번호를 붙여 보내고 번호로 돌려받는다 — 안건 이름(topic)은 두 항목이 같을 수 있어
    짝을 맞추는 열쇠로 쓸 수 없다. 근거 한 문장만으로는 무엇을 채울 수 있는지 안 보이므로
    앞뒤 대화(context_around)를 함께 주고, 이미 쓴 다른 결정 문장도 함께 줘서 같은 문장을
    다시 쓰지 않게 한다.
    """
    blocks = []
    for order, index in enumerate(flagged, start=1):
        item = items[index]
        lines = [
            "%d) 안건: %s" % (order, item["topic"]),
            "   지금 문장: %s" % item["candidates"][0]["text"],
            "   고쳐야 할 점: %s" % " / ".join(item["check"]["reasons"]),
            "   원문 근거: %s" % item["evidence"],
        ]
        context = decision_compiler.context_around(source, item["evidence"])
        if context:
            lines.append("   근거 앞뒤 대화: %s" % context)
        others = _other_sentences(items, index)
        if others:
            lines.append("   이 회의의 다른 결정 문장(겹치면 안 된다): %s" % " | ".join(others))
        blocks.append("\n".join(lines))
    return (
        "아래는 회의 녹취에서 뽑은 결정들에 대해 네가 쓴 '그대로 공개될 확정 문장'인데, 그대로 쓸 수 없는 "
        "문장들이다. 항목마다 적힌 '고쳐야 할 점'을 해결해서 다시 써라.\n"
        "- 원문 근거를 그대로 옮겨 쓰지 마라. 누가·무엇을·언제까지가 들어간 실행 문장으로 써야 한다.\n"
        "- 빠진 담당·기한·범위는 회의 정보와 앞뒤 대화에서 가장 그럴듯한 값으로 채워라.\n"
        "- 같은 회의의 다른 결정 문장과 겹치지 않게 써라.\n"
        "- \"나중에\", \"각자\", \"적당히\", \"그쪽\"처럼 사람마다 다르게 읽는 표현을 쓰지 마라.\n"
        + decision_compiler._meeting_info_block(meeting_info) +
        "\n고칠 항목:\n" + "\n\n".join(blocks) + "\n\n"
        "반드시 아래 JSON 배열 형식으로만 답하라. n은 위 항목 번호다. 채운 자리는 fills에 남겨라"
        "(gap은 기한|담당|범위|선택지, source는 원문|회의정보|AI추정, source가 \"원문\"이면 quote에 전사본 구절을 그대로):\n"
        '[{"n": 1, "candidate": "다시 쓴 확정 문장", '
        '"fills": [{"gap": "기한", "value": "12월 5일", "source": "AI추정", "quote": ""}]}]\n'
    )


def _parse_refined(raw, source, meeting_info):
    """재생성 응답 → {항목 번호: {"text", "fills"}}. 읽을 수 없으면 빈 dict.

    읽지 못한 번호는 호출자가 '원본 유지'로 처리한다 — 한 항목의 형식이 틀렸다고 나머지를 버리지 않는다.
    """
    data = decision_compiler._json_array(raw)
    if not data:
        return {}
    rewritten = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            number = int(row.get("n"))
        except (TypeError, ValueError):
            continue
        text = str(row.get("candidate", "")).strip()
        if not text:
            continue
        rewritten[number] = {
            "text": text,
            "fills": decision_compiler.parse_fills(row, source, meeting_info),
        }
    return rewritten


def _regenerate(items, flagged, source, meeting_info):
    """걸린 항목 전부를 한 번의 호출로 다시 쓴다. 실패하면 빈 dict.

    decision_compiler._generate_parsed를 쓰지 않는 이유: 그쪽은 형식이 끝까지 안 맞으면
    DecisionCompileError를 올려 회의 추출 자체를 실패시킨다. 재생성은 덧붙이는 단계이므로
    실패해도 추출 결과는 그대로 살려야 한다.
    """
    try:
        raw = services._generate_clean(_refine_prompt(items, flagged, source, meeting_info))
    except OSError:
        return {}
    if raw is None:
        return {}
    return _parse_refined(raw, source, meeting_info)


def _discard(item, text):
    """추천 문장을 버리고 '직접 쓰기' 흐름으로 보낸다.

    빈칸에서 시작하지 않게 마지막 시도 문장을 draft_hint로 남긴다 — 모호어가 섞여 있으면 그
    자리만 빈칸 표시가 된다. 채운 값도 함께 버린다(없는 문장에 색을 칠할 수는 없다).
    """
    item["candidates"] = []
    item["fills"] = []
    item["draft_hint"] = item["draft_hint"] or decision_compiler._draft_hint_text(text)


def refine_candidates(items, source, meeting_info=None):
    """추천 문장을 검증하고, 걸린 것만 한 번 다시 쓰게 하고, 그래도 걸리면 버린다.

    같은 items를 고쳐서 돌려준다. 검사한 항목에는 item["check"]가 붙는다:
    {"reasons": [걸린 이유], "regenerated": 다시 써서 통과, "discarded": 버림,
     "kept": 재생성 실패로 원본 유지}
    걸린 항목이 하나도 없으면 LLM을 부르지 않는다.
    """
    participants = (meeting_info or {}).get("participants") or ()
    flagged = []
    for index, item in enumerate(items):
        if not _checkable(item):
            continue
        passed, reasons = _check(item, items, index, participants)
        item["check"] = {"reasons": reasons, "regenerated": False, "discarded": False, "kept": False}
        if not passed:
            flagged.append(index)
    if not flagged:
        return items

    rewritten = _regenerate(items, flagged, source, meeting_info)
    for order, index in enumerate(flagged, start=1):
        item = items[index]
        check = item["check"]
        original = item["candidates"][0]["text"]
        candidate = rewritten.get(order)
        if candidate is None:
            check["kept"] = True
            logger.info(
                "추천 문장 검증: '%s' — %s / 재생성 실패로 원본 유지",
                item["topic"], " / ".join(check["reasons"]),
            )
            continue
        item["candidates"][0] = {"text": candidate["text"], "evidence": "", "recommended": True}
        item["fills"] = candidate["fills"]
        passed, reasons = _check(item, items, index, participants)
        vague = _vague_reason(candidate["text"])
        if passed and not vague:
            check["regenerated"] = True
            logger.info(
                "추천 문장 검증: '%s' — %s / 재생성 성공",
                item["topic"], " / ".join(check["reasons"]),
            )
            continue
        check["discarded"] = True
        # 두 번째 시도도 같은 이유로 걸리는 일이 많다 — 로그에 같은 문구가 두 번 찍히지 않게 합친다.
        for reason in ([vague] if vague else reasons):
            if reason not in check["reasons"]:
                check["reasons"].append(reason)
        _discard(item, candidate["text"] or original)
        logger.info(
            "추천 문장 검증: '%s' — %s / 재생성 뒤에도 걸려 폐기",
            item["topic"], " / ".join(check["reasons"]),
        )
    return items
