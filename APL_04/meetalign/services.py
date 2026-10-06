"""LLM 연결 지점. OLLAMA_MODEL 환경변수가 있을 때만 로컬 Ollama를 호출하고, 없으면 현재 동작(고정 문항, 가짜 응답)을 그대로 쓴다."""

import json
import os
import urllib.request

FAKE_CHECK_QUESTIONS = [
    "이번 회의의 핵심 결정 사항은?",
    "내가 맡은 다음 할 일은?",
]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
TIMEOUT_SECONDS = 120


def _enabled():
    return bool(os.environ.get("OLLAMA_MODEL"))


def _generate(prompt):
    body = json.dumps({"model": os.environ["OLLAMA_MODEL"], "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))["response"].strip()


def check_questions(meeting):
    if not (_enabled() and meeting.record):
        return list(FAKE_CHECK_QUESTIONS)
    prompt = (
        "다음 회의록을 읽고, 팀원들이 서로 다르게 이해했을 수 있는 핵심 질문을 한 줄에 하나씩 2개만 써라. 반드시 한국어로만 쓰고 다른 언어를 섞지 마라. "
        "번호나 설명 없이 질문만 써라.\n\n회의록:\n" + meeting.record
    )
    try:
        lines = [line.strip() for line in _generate(prompt).splitlines() if line.strip()]
    except OSError:
        return list(FAKE_CHECK_QUESTIONS)
    return lines[:2] or list(FAKE_CHECK_QUESTIONS)


def chat_reply(meeting, text):
    if not (_enabled() and meeting.record):
        return "(가짜 LLM 응답) '%s'에 대한 답변입니다." % text
    prompt = (
        "아래 회의록 내용만 근거로 질문에 답하라. 반드시 한국어로만 답하고 다른 언어를 섞지 마라. 회의록에 없는 내용이면 '회의록에 없는 내용입니다'라고 답하라.\n\n"
        "회의록:\n" + meeting.record + "\n\n질문: " + text
    )
    try:
        return _generate(prompt)
    except OSError:
        return "(LLM 연결 실패: Ollama가 실행 중인지 확인하세요)"


def is_consistent(texts):
    return len({t.strip() for t in texts}) <= 1
