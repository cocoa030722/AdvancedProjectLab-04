"""LLM 연결 지점. 제공자가 정해지기 전까지는 현재 동작을 그대로 돌려준다."""

FAKE_CHECK_QUESTIONS = [
    "이번 회의의 핵심 결정 사항은?",
    "내가 맡은 다음 할 일은?",
]


def check_questions(meeting):
    return list(FAKE_CHECK_QUESTIONS)


def chat_reply(meeting, text):
    return "(가짜 LLM 응답) '%s'에 대한 답변입니다." % text


def is_consistent(texts):
    return len({t.strip() for t in texts}) <= 1
