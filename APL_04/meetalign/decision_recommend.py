"""AI 추천 문장 만들기 — 결정 추출과 따로 LLM을 한 번 더 부른다.

예전에는 한 번의 호출로 [실행 항목 추출 · 모호 판정 · 추천 문장 · 채운 값]을 다 받았다. 그러면 모델이
추출과 판정에 주의를 쓰느라 추천 문장은 원문을 말만 바꿔 되풀이하거나 다른 안건과 같은 틀을 반복했다.
그래서 둘로 나눴다.

1. 추출(decision_compiler._compile_prompt): 실행 항목을 뽑고 모호 여부·사유·영향·원문 근거를 판정한다.
   명확한 항목은 원문에 나온 값만으로 확정 문장(final)을 쓴다.
2. 추천(이 모듈): 확인이 필요하다고 판정된 항목만 모아, 항목마다 '그대로 공개될 확정 문장' 하나와
   채운 값(fills)을 쓴다. 이 호출은 추천 문장 하나에만 집중하고, 다음을 함께 본다.
   - 회의 정보(날짜·참석자·마감일·이월 안건) — 빈칸을 채울 근거
   - 항목별 원문 근거와 앞뒤 대화 — 무엇이 빠졌는지
   - 이미 확정된 결정 문장 — 겹치거나 기한·담당이 모순되지 않게
   - 같이 쓰는 다른 추천 문장 — 한 번에 쓰므로 서로 맞춰 쓰게 한다
   - 전사본 전체 — 회의 앞부분의 제약(가능·불가능한 시간 등)을 반영하게

추천 호출이 실패해도(연결 끊김·읽을 수 없는 응답) 추출 결과는 살린다. 추천 문장이 없는 항목은 진행자가
'직접 쓰기'나 '다음 회의로 넘김'으로 처리한다 — 회의 전체를 실패시키는 것보다 손해가 작다.

이 뒤에 decision_refine이 추천 문장을 규칙으로 되짚고, 걸린 것만 한 번 더 다시 쓰게 한다.
"""

import logging

from . import decision_compiler, services

logger = logging.getLogger(__name__)

# 근거 앞뒤로 몇 문장을 더 보여 줄지. 추출 화면(진행자)보다 넓게 준다 — 빈칸을 채울 단서가 근거 바로
# 옆이 아니라 몇 마디 앞뒤에 나오는 일이 많다.
CONTEXT_WINDOW = 2


def _targets(items):
    """추천 문장을 쓸 항목의 인덱스 — AI가 모호하다고 판정한 항목만."""
    return [index for index, item in enumerate(items) if item["ambiguous"]]


def _settled_sentences(items):
    """이미 확정된(명확) 결정 문장 — 추천 문장이 겹치거나 모순되면 안 되는 기준."""
    return [item["final"] for item in items if not item["ambiguous"] and item["final"]]


def recommend_prompt(items, targets, source, meeting_info=None):
    """확인이 필요한 항목들의 추천 문장을 한 번에 쓰게 하는 프롬프트.

    항목마다 번호를 붙여 보내고 번호로 돌려받는다 — 안건 이름(topic)은 두 항목이 같을 수 있다.
    """
    blocks = []
    for number, index in enumerate(targets, start=1):
        item = items[index]
        lines = ["%d) 안건: %s" % (number, item["topic"])]
        if item["why"]:
            lines.append("   확인이 필요한 이유: %s" % item["why"])
        if item["impact"]:
            lines.append("   그대로 두면 생길 일: %s" % item["impact"])
        if item["evidence"]:
            lines.append("   원문 근거: %s" % item["evidence"])
            context = decision_compiler.context_around(source, item["evidence"], window=CONTEXT_WINDOW)
            if context:
                lines.append("   근거 앞뒤 대화: %s" % context)
        blocks.append("\n".join(lines))
    settled = _settled_sentences(items)
    settled_block = (
        "\n이미 확정된 결정(추천 문장이 이것과 겹치거나 기한·담당·장소가 모순되면 안 된다):\n"
        + "\n".join("- %s" % sentence for sentence in settled) + "\n"
    ) if settled else ""
    return (
        "아래는 회의 녹취에서 뽑은 안건 가운데 팀원들이 다르게 이해하거나 다르게 행동할 수 있어서 진행자의 확인이 "
        "필요한 것들이다. 항목마다 진행자에게 가장 추천하는 '그대로 공개될 확정 문장' 하나를 써라.\n"
        + decision_compiler._meeting_info_block(meeting_info) + settled_block +
        "\n확정 문장 규칙: 그 문장이 팀원에게 그대로 공개된다. 누가·무엇을·언제까지가 문장 안에 들어간 실행 "
        "문장으로 쓴다. \"~라는 뜻\", \"A일 수도 B일 수도\" 같은 해설체로 쓰지 마라. 어느 쪽이 맞는지 해설하지 "
        "마라 — 확정하는 것은 회의 진행자다. \"쯤\", \"나중에\", \"각자\", \"적당히\", \"그쪽\", \"그분\"처럼 "
        "사람마다 다르게 읽는 표현을 쓰지 마라.\n"
        + decision_compiler.FILL_RULES +
        "  결론을 다음으로 미루기로 한 안건도 원문 흐름상 가장 그럴듯한 쪽을 골라 쓰고, 고른 값은 fills에 "
        "\"AI추정\"으로 남겨라.\n"
        "  전사본 전체에서 각자 말한 가능·불가능한 시간과 사정을 먼저 모아 두고, 추천 문장이 그것과 부딪히지 "
        "않게 써라. 함께 쓰는 추천 문장끼리도 기한·담당·순서가 서로 맞아야 한다(예: 자료를 받아야 시작하는 "
        "일의 기한은 자료 공유 기한보다 뒤여야 한다).\n"
        "\n추천 문장을 쓸 항목:\n" + "\n\n".join(blocks) + "\n\n"
        "반드시 아래 JSON 배열 형식으로만 답하라. 설명이나 코드블록 표시 없이 JSON만 출력하라. n은 위 항목 "
        "번호이고, 항목마다 하나씩 빠짐없이 답하라:\n"
        '[{"n": 1, "candidate": "가장 추천하는 확정 문장", '
        '"fills": [{"gap": "기한", "value": "12월 5일", "source": "AI추정", "quote": ""}]}]\n'
        "\n전사본:\n" + source
    )


def parse_recommendations(raw, items, targets, source=None, meeting_info=None):
    """추천 응답 → {항목 인덱스: {"candidates", "draft_hint", "fills"}}. 읽을 수 없으면 빈 dict.

    번호(n)로 짝을 맞추고, 번호가 없으면 안건 이름(topic)으로 맞춘다 — 모델이 번호 대신 안건 이름을
    되돌려 주는 일이 있다. 안건 이름이 겹치면 앞쪽 항목부터 채운다.

    모호어·지시어가 남은 문장은 추출 때와 같은 기준(_parse_candidates)으로 버리고 '직접 쓰기'
    참고 문구(draft_hint)로만 남긴다.
    """
    data = decision_compiler._json_array(raw)
    if not data:
        return {}
    by_number = {number: index for number, index in enumerate(targets, start=1)}
    unclaimed_by_topic = {}
    for index in targets:
        unclaimed_by_topic.setdefault(items[index]["topic"], []).append(index)
    results = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        index = None
        try:
            index = by_number.get(int(row.get("n")))
        except (TypeError, ValueError):
            pass
        if index is None:
            waiting = unclaimed_by_topic.get(str(row.get("topic", "")).strip(), [])
            index = next((i for i in waiting if i not in results), None)
        if index is None or index in results:
            continue
        raw_candidate = row.get("candidate") if row.get("candidate") is not None else row.get("candidates")
        candidates, draft_hint = decision_compiler._parse_candidates(raw_candidate)
        if not candidates and not draft_hint:
            continue
        results[index] = {
            "candidates": candidates,
            "draft_hint": draft_hint,
            "fills": decision_compiler.parse_fills(row, source, meeting_info) if candidates else [],
        }
    return results


def recommend_candidates(items, source, meeting_info=None):
    """확인이 필요한 항목의 추천 문장을 별도 호출로 만들어 items에 채운다. 같은 items를 돌려준다.

    응답 형식이 맞지 않으면 같은 프롬프트로 한 번 더 보낸다(decision_compiler.PARSE_RETRIES). 끝내 못
    받은 항목은 추천 문장 없이 둔다. 확인이 필요한 항목이 없으면 LLM을 부르지 않는다.
    """
    targets = _targets(items)
    if not targets:
        return items
    prompt = recommend_prompt(items, targets, source, meeting_info)
    results = {}
    for _ in range(decision_compiler.PARSE_RETRIES + 1):
        try:
            raw = services._generate_clean(prompt)
        except OSError:
            logger.warning("추천 문장 호출 실패(연결) — 추천 문장 없이 진행합니다")
            break
        if raw is None:
            logger.warning("추천 문장 호출 실패(응답 언어) — 추천 문장 없이 진행합니다")
            break
        results = parse_recommendations(raw, items, targets, source, meeting_info)
        if results:
            break
    for index in targets:
        item = items[index]
        found = results.get(index)
        if found is None:
            # 추출 응답에 모델이 덧붙여 보낸 문장이 있으면 그대로 둔다(예전 형식). 없으면 직접 쓰기로 간다.
            continue
        item["candidates"] = found["candidates"]
        item["draft_hint"] = found["draft_hint"] or item["draft_hint"]
        item["fills"] = found["fills"]
    missing = [items[index]["topic"] for index in targets if index not in results]
    logger.info(
        "추천 문장: 대상 %d건 / 작성 %d건%s", len(targets), len(targets) - len(missing),
        (" / 못 받음: %s" % ", ".join(missing)) if missing else "",
    )
    return items
