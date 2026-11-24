"""실시간 회의 세션 — 조각 전사, 애매함 분석, 종료 뒤 통합 녹음.

흐름
1. 브라우저가 녹음을 LIVE_CHUNK_SECONDS초 조각으로 잘라 올린다(views.live_chunk). 조각마다 바로 전사한다
   (transcribe_chunk — 화자 분리 없이, services.transcribe_live).
2. 조각 전사가 끝날 때 마지막 분석에서 ANALYZE_INTERVAL_SECONDS가 지났으면(또는 진행자가 일시정지하면), 지금까지의
   녹취 전체와 새로 들어온 부분을 함께 AI에게 보내 애매한 점을 찾는다(analyze). 앞에서 한 말과 새 말을 함께 보아야
   "앞에서 안 된다고 한 날"과의 충돌이나 뒤에서 정리된 것을 알 수 있다. 가장 최근 말 HOLDBACK_LINES개는 바로 뒤에 답이
   이어질 수 있어 판단하지 않고 [바로 뒤에 이어진 말]로만 보여 준다 — 다음 분석이 판단한다.
   AI 호출은 실시간용(services.generate_live)이라 짧게 기다리고, 실패하면 이번 분석을 건너뛴다.
3. 종료하면 조각 녹음을 순서대로 하나의 WAV로 합쳐 일반 녹음(Recording)으로 등록만 한다(finish, 상태 ready).
   진행자가 회의 화면에서 '제출하고 회의록 만들기'를 누르면 다른 업로드와 함께 기존 흐름을 탄다(화자 분리를 포함한
   통합 전사 → 회의록 → 결정 추출). 실시간 녹취는 회의 중 보기용이고, 회의록의 원문은 통합 전사본이다.

백그라운드 스레드에서 돈다(BACKGROUND_LIVE). 진행 상태는 녹음 처리(tasks)처럼 메모리에 두되, 서버가 재시작되어
멈춘 조각은 화면이 상태를 물을 때 다시 전사한다(recover).
"""

import json
import logging
import os
import re
import tempfile
import threading
import time

from django.conf import settings
from django.core.files import File
from django.db import close_old_connections, transaction
from django.utils import timezone

from . import decision_checks, decision_compiler, live_rules, services
from .models import LiveChunk, LiveFlag, LiveSession, Recording

logger = logging.getLogger(__name__)

# 조각의 최대 길이(초). 브라우저는 말이 잠깐 끊기는 지점(약 0.7초 조용함)에서 조각을 끊어 바로 올리고, 쉬지 않고 말해도
# 이 길이가 되면 끊는다(live.html). 문장 단위로 잘려 단어가 중간에 끊기지 않고, 말한 뒤 몇 초면 녹취에 보인다.
# LIVE_CHUNK_SECONDS로 조정.
LIVE_CHUNK_SECONDS = 15
# 애매함 분석 간격(초). 조각은 말 한마디(약 0.7초 쉼)마다 끊겨서 조각마다 보면 5분 회의에 40번 넘게 불렀고, 무료 등급
# 한도에 걸려 느린 모델로 넘어가면서 정작 중요한 충돌을 놓쳤다. 15초 분량(말 대여섯 마디)마다 본다 — 질문과 답이
# 한 묶음으로 들어오지 않아도 최근 말 미루기(HOLDBACK_LINES)가 뒤에 이어진 말을 함께 보여 준다. 실시간 분석은 가벼운
# 모델이라(services.generate_live) 자주 불러도 결정 추출의 한도를 깎지 않는다. 마지막 분석을 시작한 뒤 이만큼 지나야 다음 분석을 시작한다
# (일시정지하면 바로 본다). 분석이 도는 중에 쌓인 말은 다음 분석이 한꺼번에 본다. LIVE_ANALYZE_INTERVAL_SECONDS로 조정.
ANALYZE_INTERVAL_SECONDS = 15
# 판단을 미루는 최근 말 수. 질문을 하거나 안건을 막 꺼낸 순간에 보면 바로 뒤에 나올 답을 모른 채 "기한을 정해 주세요"를
# 띄웠다. 가장 최근 말 몇 개는 [바로 뒤에 이어진 말]로만 보여 주고, 그 앞까지만 판단한다. 말이 끊겨 뒤에 말이 더 오지 않아도
# 너무 오래 미루지 않게, 녹취에 들어온 지 HOLDBACK_SECONDS가 지난 말은 판단한다. LIVE_HOLDBACK_LINES·LIVE_HOLDBACK_SECONDS로 조정.
HOLDBACK_LINES = 2
HOLDBACK_SECONDS = 20
# 분석할 때 보내는 녹취 길이 상한(자). 맥락을 잃지 않도록 지금까지의 녹취를 통째로 보내되, 아주 긴 회의에서 호출이 끝없이
# 커지지 않게 이 길이를 넘으면 앞부분부터 덜어 낸다(2시간 회의가 약 9만 자). LIVE_MAX_CONTEXT_CHARS로 조정.
MAX_CONTEXT_CHARS = 100000
# 한 번에 올리는 애매한 점 수. 회의 중에는 짧게 몇 개만 짚는다 — 세세한 것은 회의 뒤 결정 확인이 다시 본다.
MAX_FLAGS_PER_ANALYSIS = 3

_lock = threading.Lock()
_active_chunks = set()
_analyzing = set()
_finishing = set()
# 세션마다 마지막으로 분석을 시작한 때(time.monotonic). 메모리에만 둔다 — 서버가 재시작되면 바로 한 번 본다.
_last_analysis = {}
# 세션마다 지금 부르는 중인 모델((time.monotonic, 이름)). 한 분석이 가벼운 Gemini 실패 → 로컬 모델처럼 여러 번 부를 수
# 있어서, 화면이 "판단 중"에 지금 어느 모델인지 보이게 한다. 시간은 분석을 시작한 때(_last_analysis)부터 이어서 센다 —
# 같은 판단이면 모델이 바뀌어도 이어지고, 다음 판단이 시작되면 0부터 다시 센다.
_current_try = {}
# 세션마다 마지막 분석이 실패한 때. 일시정지 중에는 간격 없이 바로 보지만, 실패했으면 1초마다 다시 부르지 않고 판단 딜레이
# 시간만큼 쉬었다가 다시 본다(AI 장애·한도 초과 때 호출을 되풀이하지 않게).
_last_failure = {}


def chunk_seconds():
    return int(os.environ.get("LIVE_CHUNK_SECONDS", LIVE_CHUNK_SECONDS))


def _max_context_chars():
    return max(1000, int(os.environ.get("LIVE_MAX_CONTEXT_CHARS", MAX_CONTEXT_CHARS)))


# 진행자가 화면에서 고르는 판단 딜레이 시간(초)의 범위(live.html의 슬라이더).
INTERVAL_MIN_SECONDS, INTERVAL_MAX_SECONDS, INTERVAL_STEP_SECONDS = 5, 60, 5


def analyze_interval(session=None):
    """분석 간격(초). 진행자가 그 세션에서 정했으면 그 값, 아니면 LIVE_ANALYZE_INTERVAL_SECONDS(기본 15초)."""
    if session is not None and session.analyze_interval is not None:
        return session.analyze_interval
    return max(0, int(os.environ.get("LIVE_ANALYZE_INTERVAL_SECONDS", ANALYZE_INTERVAL_SECONDS)))


def clamp_interval(seconds):
    """화면에서 보낸 값을 슬라이더 범위와 단위에 맞춘다."""
    seconds = min(INTERVAL_MAX_SECONDS, max(INTERVAL_MIN_SECONDS, seconds))
    return int(round(seconds / INTERVAL_STEP_SECONDS) * INTERVAL_STEP_SECONDS)


def holdback_lines():
    return max(0, int(os.environ.get("LIVE_HOLDBACK_LINES", HOLDBACK_LINES)))


def holdback_seconds(session=None):
    """최근 말을 미루는 최대 시간. 판단 딜레이 시간보다 길게 미루지 않는다 — 딜레이를 10초로 정했는데 최근 말을 20초씩 붙잡아
    정한 시간보다 늦게 판단했다."""
    seconds = max(0, int(os.environ.get("LIVE_HOLDBACK_SECONDS", HOLDBACK_SECONDS)))
    interval = analyze_interval(session) if session is not None else 0
    # 간격 0(환경 변수로만 둘 수 있다)은 "분석 사이를 띄우지 않는다"는 뜻이지 최근 말을 미루지 말라는 뜻이 아니다.
    return min(seconds, interval) if interval > 0 else seconds


def _run(target, *args):
    """BACKGROUND_LIVE면 스레드로, 아니면(테스트) 그 자리에서 돈다."""
    if not getattr(settings, "BACKGROUND_LIVE", True):
        target(*args)
        return

    def run():
        try:
            target(*args)
        finally:
            close_old_connections()

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# 녹취
# ---------------------------------------------------------------------------


def session_transcript(session):
    """전사가 끝난 조각을 순번대로 이은 실시간 녹취."""
    return " ".join(
        chunk.transcript for chunk in session.chunks.filter(status=LiveChunk.STATUS_DONE) if chunk.transcript
    ).strip()


def _transcribe_chunk(chunk_id):
    try:
        chunk = LiveChunk.objects.select_related("session").get(pk=chunk_id)
    except LiveChunk.DoesNotExist:
        return
    try:
        text = services.transcribe_live(chunk.file.path)
        chunk.transcript = text or ""
        chunk.status = LiveChunk.STATUS_DONE
    except Exception:
        logger.exception("live chunk transcription failed (chunk %s)", chunk_id)
        chunk.status = LiveChunk.STATUS_FAILED
    chunk.save(update_fields=["transcript", "status"])
    with _lock:
        _active_chunks.discard(chunk_id)
    maybe_analyze(chunk.session)


def transcribe_chunk(chunk, silent=False):
    """조각 하나를 전사한다. silent면(브라우저가 소리가 거의 없다고 판단한 조각) 전사하지 않고 빈 녹취로 끝낸다 —
    Whisper가 조용한 조각에서 말을 지어내지 않게 하고, 전사 호출도 아낀다. 녹음 파일은 남겨 종료 뒤 합칠 때 쓴다."""
    if silent:
        LiveChunk.objects.filter(pk=chunk.pk).update(status=LiveChunk.STATUS_DONE, transcript="")
        maybe_analyze(chunk.session)
        return
    with _lock:
        if chunk.id in _active_chunks:
            return
        _active_chunks.add(chunk.id)
    _run(_transcribe_chunk, chunk.id)


def recover(session):
    """서버가 재시작되어 멈춘 조각(대기 중인데 아무도 전사하지 않는 것)을 다시 전사한다."""
    with _lock:
        active = set(_active_chunks)
    for chunk in session.chunks.filter(status=LiveChunk.STATUS_PENDING):
        if chunk.id not in active:
            transcribe_chunk(chunk)


# ---------------------------------------------------------------------------
# 애매함 분석
# ---------------------------------------------------------------------------


# 빠진 조각 순번을 기다리는 최대 시간(초). 브라우저가 조각 하나를 세 번 올려 보다 포기하면 그 순번은 끝내 오지 않는다 — 그
# 뒤 조각이 들어온 지 이만큼 지났으면 빠진 순번은 건너뛴다. 건너뛰지 않으면 그 뒤 녹취의 분석과 회의 기록 표시가 회의가
# 끝날 때까지 멈췄다. 재전송은 몇 초 안에 끝나므로 넉넉하게 둔다.
GAP_GRACE_SECONDS = 30


def contiguous_until(chunks, last):
    """last 다음부터 빈틈없이 전사가 끝난(또는 실패한) 마지막 조각 순번. chunks는 순번순이고 last보다 뒤의 것이다. 앞 조각이
    아직 전사 중이면 그 뒤는 기다린다 — 순서가 뒤섞인 녹취로 판단하거나 보여 주지 않게. 빠진 순번은 GAP_GRACE_SECONDS 동안만
    기다린다(views._lines_until도 같은 규칙)."""
    now = timezone.now()
    for chunk in chunks:
        if chunk.status == LiveChunk.STATUS_PENDING:
            break
        if chunk.seq != last + 1 and (now - chunk.created_at).total_seconds() < GAP_GRACE_SECONDS:
            break
        last = chunk.seq
    return last


def _ready_seq(session):
    """analyzed_seq 다음부터 빈틈없이 전사가 끝난(또는 실패한) 마지막 조각 순번(contiguous_until)."""
    return contiguous_until(session.chunks.filter(seq__gt=session.analyzed_seq).order_by("seq"), session.analyzed_seq)


def _settled_seq(session, ready):
    """판단해도 되는 마지막 조각 순번. 가장 최근 말 holdback_lines()개는 뒤에 답이 이어질 수 있어 미룬다. 다만 녹취에
    들어온 지 holdback_seconds()가 지난 말은 뒤에 말이 더 오지 않았어도 판단한다(말이 끊긴 채 오래 기다리지 않게)."""
    lines, seconds = holdback_lines(), holdback_seconds(session)
    now = timezone.now()
    spoken_after = 0
    for chunk in session.chunks.filter(seq__gt=session.analyzed_seq, seq__lte=ready).order_by("-seq"):
        if spoken_after >= lines or (now - chunk.created_at).total_seconds() >= seconds:
            return chunk.seq
        if chunk.transcript:
            spoken_after += 1
    return session.analyzed_seq


def maybe_analyze(session, force=False):
    """판단할 녹취가 생겼고 마지막 분석에서 analyze_interval()초가 지났으면 분석을 시작한다. 이미 분석 중이면 넘어간다.

    force거나 일시정지 중이면 간격을 기다리지 않고, 최근 말도 미루지 않고 지금까지 쌓인 것을 모두 본다 — 쉬는 동안 정리할 수
    있게. 일시정지를 누른 때는 마지막 조각이 아직 올라가거나 전사되는 중이라, 그 조각은 전사가 끝난 뒤에 이 규칙으로 본다.
    다만 마지막 분석이 실패했으면 판단 딜레이 시간만큼은 쉬었다가 다시 본다.
    """
    session.refresh_from_db(fields=["analyzed_seq", "status", "analyze_interval"])
    if session.status == LiveSession.STATUS_ENDED:
        # 끝난 회의는 통합 전사본으로 결정 추출이 다시 본다 — 회의 중 알림용 분석을 더 돌릴 이유가 없다.
        return
    force = force or session.status == LiveSession.STATUS_PAUSED
    ready = _ready_seq(session)
    until = ready if force else _settled_seq(session, ready)
    if until <= session.analyzed_seq:
        return
    if not services.live_enabled():
        return
    with _lock:
        if session.id in _analyzing:
            return
        last, failed = _last_analysis.get(session.id), _last_failure.get(session.id)
        if not force and last is not None and time.monotonic() - last < analyze_interval(session):
            return
        if failed is not None and time.monotonic() - failed < analyze_interval(session):
            return
        _analyzing.add(session.id)
        _last_analysis[session.id] = time.monotonic()
    _run(_analyze, session.id, until, ready)


# 실시간에만 더 보는 모호 표현 — 결정 추출의 힌트 목록(AMBIGUOUS_PHRASES 등)에 점검표 1·2·6번의 예를 더한다. 가벼운
# 모델이 "최대한 빨리", "일단 알겠어요"를 자주 넘겼다. 판정은 AI가 한다(검토하라고 짚기만 한다).
LIVE_HINT_PHRASES = ("최대한 빨리", "되는 대로", "되는대로", "봐서", "이따", "단톡에서", "일단 알겠", "아마", "해 볼게", "해볼게")


def _flag_line(flag, note="", numbered=False):
    """numbered면 번호(#id)를 붙인다 — 아직 열린 것은 AI가 '정해졌다'(settled)고 답할 때 이 번호로 가리킨다."""
    number = "#%s " % flag.id if numbered and getattr(flag, "id", None) else ""
    topic = "[%s] " % flag.topic if getattr(flag, "topic", "") else ""
    return "- %s%s%s: %s%s" % (number, topic, flag.phrase, flag.reason, note)


def _resolved_note(flag):
    return " (회의에서 정해짐)" if getattr(flag, "resolved_auto", False) else " (진행자가 처리함)"


def analysis_status(session):
    """화면에 보일 애매함 분석 상태 → {"state": ..., "model": 이름}. 세 단계로만 나눈다(세부 단계와 남은 초를 보였더니 알아보기
    어려웠다).

    - busy: 판단 중. model은 지금 부르는 모델(화면에서 마우스를 올리면 보인다).
    - waiting: 다음 판단까지 대기 중 — 아직 판단하지 않은 말이 있다(간격·최근 말 미루기·전사·실패 뒤 다시 시도를 기다리는 중).
      seconds는 다음 판단까지 남은 초다. 다음 판단은 판단 딜레이 시간이 차고 최근 말 미루기가 끝나야 시작하므로 둘 중 긴 쪽,
      실패했으면 다시 시도까지다. 일시정지 중이거나 아직 글로 옮기는 중이라 알 수 없으면 0(화면에 초를 붙이지 않는다).
    - idle: 입력이 없음 — 판단할 새 말이 없다(일시정지 중 포함).
    - off: 끝난 세션이거나 AI가 설정되지 않았다(화면에 보이지 않는다).
    """
    # 바로 앞에서 분석이 끝났을 수 있다(views.live_state가 묻기 직전에 maybe_analyze를 부른다) — 최신 값으로 본다.
    session.refresh_from_db(fields=["analyzed_seq", "status", "analyze_interval"])
    if session.status == LiveSession.STATUS_ENDED or not services.live_enabled():
        return {"state": "off", "seconds": 0, "model": ""}
    with _lock:
        busy = session.id in _analyzing
        current = _current_try.get(session.id)
        last, failed = _last_analysis.get(session.id), _last_failure.get(session.id)
    if busy:
        return {"state": "busy", "seconds": 0, "model": current[1] if current else ""}
    ready = _ready_seq(session)
    if ready <= session.analyzed_seq:
        transcribing = session.chunks.filter(seq__gt=session.analyzed_seq, status=LiveChunk.STATUS_PENDING).exists()
        return {"state": "waiting" if transcribing else "idle", "seconds": 0, "model": ""}
    return {"state": "waiting", "seconds": _seconds_to_next(session, ready, last, failed), "model": ""}


def _seconds_to_next(session, ready, last, failed):
    """다음 판단까지 남은 초(올림). maybe_analyze가 보는 조건을 그대로 따른다."""
    clock, interval = time.monotonic(), analyze_interval(session)
    if failed is not None and clock - failed < interval:
        return _ceil(interval - (clock - failed))
    if session.status == LiveSession.STATUS_PAUSED:
        return 0  # 일시정지 중에는 간격·최근 말 미루기 없이 바로 본다
    left = 0 if last is None else interval - (clock - last)
    if _settled_seq(session, ready) <= session.analyzed_seq:
        oldest = (session.chunks.filter(seq__gt=session.analyzed_seq, seq__lte=ready)
                  .exclude(transcript="").order_by("seq").first())
        if oldest is not None:
            held = holdback_seconds(session) - (timezone.now() - oldest.created_at).total_seconds()
            left = max(left, held)
    return _ceil(left)


def _ceil(seconds):
    return max(0, int(seconds + 0.999))


def analyze_prompt(new_text, before_text, open_flags, resolved_flags, trimmed=False, after_text="", dismissed_flags=(),
                   hint_lines=()):
    """before_text는 지금까지(이미 본) 녹취 전체, new_text는 이번에 판단할 녹취, after_text는 그 바로 뒤에 이어진 말이다
    (판단을 미룬 최근 말 — 참고만 한다). dismissed_flags는 진행자가 잘못 짚었다고 지운 것, hint_lines는 새로 들어온 녹취
    가운데 모호 표현이 든 말이다(decision_compiler.review_hint_lines).

    판단 기준은 회의 뒤 결정 추출과 같은 점검표(decision_compiler.AMBIGUITY_CHECKLIST)와 "행동이 갈릴 때만" 기준이다. 결정
    추출처럼 안건 단위로 보되, 회의 중이라 아직 논의가 이어지는 안건은 결론이 나올 때까지 기다린다.
    """
    lines = []
    if open_flags or resolved_flags:
        listed = ([_flag_line(f, numbered=True) for f in open_flags]
                  + [_flag_line(f, _resolved_note(f)) for f in resolved_flags])
        lines.append("[이미 올린 애매한 점 — 같은 안건의 같은 문제는 표현이 달라도 다시 올리지 마라]\n" + "\n".join(listed))
    if dismissed_flags:
        lines.append("[진행자가 애매하지 않다고 지운 것 — 같은 문제나 같은 이유로는 다시 올리지 마라]\n"
                     + "\n".join(_flag_line(f) for f in dismissed_flags))
    if before_text:
        lines.append("[지금까지의 녹취%s]\n" % (" (앞부분 일부 생략)" if trimmed else "") + before_text)
    lines.append("[새로 들어온 녹취]\n" + new_text)
    if hint_lines:
        lines.append("[새로 들어온 녹취에서 모호한 표현이 든 말 — 하나씩 반드시 검토하라. 검토해서 문제가 아니면 올리지 "
                     "않아도 된다]\n" + "\n".join("- %s" % line for line in hint_lines))
    if after_text:
        lines.append("[바로 뒤에 이어진 말 — 참고만 하고 여기서는 phrase를 고르지 마라]\n" + after_text)
    return (
        "다음은 진행 중인 회의의 실시간 녹취다. 팀원들이 회의를 마친 뒤 각자 움직이려면 무엇을 알아야 하는지 기준으로, "
        "새로 들어온 녹취가 건드린 실행 항목(확정된 결정, 누군가 하겠다고 한 일, 정하려다 미룬 것, 다음 회의의 날짜·"
        "시간·장소)을 안건별로 살피고, 팀원들이 다르게 이해하거나 다르게 행동할 수 있게 된 애매한 점을 회의 중에 바로 "
        "짚어 줘라. 지금까지의 녹취 전체로 맥락을 잡는다. 새 말이 앞에서 한 말과 부딪히게 됐다면 그것도 짚고, 그때 phrase는 "
        "앞 녹취에 나온 표현이어도 된다. 새 말로 앞의 애매함이 풀렸다면 올리지 않는다. 회의를 멈추지 않고 한눈에 읽을 수 "
        "있게 짧게 쓴다 — 세세한 판단은 회의가 끝난 뒤 따로 한다.\n\n"
        "점검표(회의 뒤 결정 확인과 같은 기준):\n"
        + decision_compiler.AMBIGUITY_CHECKLIST +
        "점검표에 해당하더라도 그 차이 때문에 팀원의 담당·실행 시점·범위·완료 기준이 어떻게 갈리는지 설명할 수 있을 때만 "
        "올린다. 표현이 느슨해도 실제로 할 일이 같으면 올리지 않는다. 정하려다 미룬 것은 행동 차이가 아직 없어도 올린다.\n\n"
        "회의 중이라 아직 논의가 이어지는 안건이 많다. 이런 것은 올리지 마라:\n"
        "- 질문·제안이나 안건을 막 꺼낸 말 자체(\"언제까지 해 주실 수 있어요?\", \"역할을 나눠야 하는데\", \"다들 언제 "
        "괜찮으세요?\") — 그 답이나 결론이 애매할 때 짚는다. 다만 제안이 앞에서 누군가 말한 사정과 부딪히면 바로 짚는다.\n"
        "- [바로 뒤에 이어진 말]에서 답이 나왔거나 정리된 것.\n"
        "- 이미 올린 애매한 점과 같은 안건의 같은 문제. 같은 안건에서 정해야 할 것이 같으면 표현이 달라도 같은 문제다(예: "
        "자료조사 기한을 이미 짚었으면 \"최대한 빨리\", \"대충이라도 나오면\", \"찾는 대로\"는 다시 올리지 않는다). 같은 "
        "안건이라도 정해야 할 것이 다르면(기한을 짚었는데 담당자가 비었다, 날짜를 짚었는데 누군가의 사정과 겹친다) 새로 "
        "올린다.\n"
        "- 인사·자기소개·진행 멘트·잡담, 녹취 안에서 이미 정리된 것, 새로 들어온 녹취와 상관없는 앞부분의 애매함(이미 앞에서 "
        "살펴봤다).\n"
        "확실한 것만 최대 %d개 올리고, 없으면 빈 목록으로 둔다.\n"
        "또 [이미 올린 애매한 점] 가운데 번호(#)가 붙은 아직 열린 것이 [새로 들어온 녹취]에서 분명하게 정해졌으면 settled에 "
        "그 번호(id)와, 정한 말을 [새로 들어온 녹취]에서 토씨 하나 틀리지 않고 옮긴 quote를 적는다. 무엇으로 정했는지(누가·"
        "언제·어디서·무엇을) 녹취에서 알 수 있을 때만 넣는다 — \"알겠어요\"처럼 확답이 아니거나, 다시 미뤘거나, 일부만 "
        "정해졌으면 넣지 않는다. 없으면 빈 목록으로 둔다.\n\n" % MAX_FLAGS_PER_ANALYSIS
        + "\n\n".join(lines) + "\n\n"
        "순서대로 판단하라. 먼저 constraints에 [지금까지의 녹취]와 [새로 들어온 녹취]에서 누군가 안 된다고 하거나 사정이 "
        "있다고 한 것(누가·언제·무엇)을 빠짐없이 적는다. 그다음 [새로 들어온 녹취]의 제안·결정·대답을 constraints와 하나씩 "
        "맞대 보고, 겹치면 반드시 flags에 올린다(말한 사람이 같아도 올린다). 마지막으로 점검표의 나머지 항목을 본다.\n"
        "반드시 아래 JSON 객체 하나로만 답하라. 설명이나 코드블록 표시 없이 JSON만 출력하라.\n"
        "topic은 어느 안건의 무슨 문제인지 12자 안팎으로 쓴다(예: \"자료조사 기한\", \"다음 회의 시간\", \"회의 장소\"). "
        "phrase는 [지금까지의 녹취]나 [새로 들어온 녹취]에 나온 표현을 토씨 하나 틀리지 않고 짧게 그대로 옮긴다. reason은 "
        "지금 회의에서 무엇을 정하면 되는지 알 수 있게 40자 안팎의 한 문장으로 쓰고, 점검표 번호는 쓰지 마라(예: \"몇 "
        "요일까지인지 정해 주세요\").\n"
        '{"constraints": [{"who": "민혁", "when": "목요일 3시~4시 반", "what": "전공 수업"}], '
        '"flags": [{"topic": "다음 회의 시간", "phrase": "목요일 오후 4시", "reason": "민혁 씨 수업 시간과 겹치니 다시 정해 주세요"}, '
        '{"topic": "PPT 기한", "phrase": "다음 주쯤", "reason": "몇 요일까지인지 정해 주세요"}], '
        '"settled": [{"id": 3, "quote": "장소는 도서관 스터디룸으로 할게요"}]}'
    )


def _json_object(raw):
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def parse_settled(raw, new_text, open_flags):
    """분석 응답의 settled → [(열린 LiveFlag, quote)]. 아직 열린 것(처리 안 됨·안 지움·자동 확인을 되돌린 적 없음)만,
    quote가 이번에 판단한 녹취에 실제로 있을 때만 받는다 — 가벼운 모델이 지어낸 근거로 애매한 점을 닫지 않게."""
    data = _json_object(raw) or {"settled": _array_field(raw, "settled") or []}
    candidates = {f.id: f for f in open_flags if not f.resolved and not f.dismissed and not f.settled_quote}
    squashed_new = decision_checks.squash_transcript(new_text)
    settled, taken = [], set()
    for row in data.get("settled") or []:
        if not isinstance(row, dict):
            continue
        try:
            flag_id = int(str(row.get("id", "")).lstrip("#"))
        except ValueError:
            continue
        quote = " ".join(str(row.get("quote", "")).split())[:300]
        flag = candidates.get(flag_id)
        if flag is None or flag_id in taken or not quote or not decision_checks.quote_in_source(quote, squashed_new):
            continue
        taken.add(flag_id)
        settled.append((flag, quote))
    return settled


def _array_field(raw, name):
    """전체 JSON이 깨졌을 때 name 배열만 꺼낸다. 로컬 작은 모델이 constraints를 닫으면서 객체까지 닫아
    ({"constraints": [...]}, "flags": [...]}) 응답 전체를 못 읽고 맞는 지적까지 버리곤 했다."""
    match = re.search(r'"%s"\s*:\s*(?=\[)' % name, raw or "")
    if not match:
        return None
    try:
        rows, _ = json.JSONDecoder().raw_decode(raw, match.end())
    except (json.JSONDecodeError, ValueError):
        return None
    return rows if isinstance(rows, list) else None


def _flags_only(raw):
    flags = _array_field(raw, "flags")
    return {"flags": flags} if flags is not None else None


def _reason_key(reason):
    """사유 비교용 — 띄어쓰기와 문장부호(끝의 마침표 등)는 보지 않는다."""
    return re.sub(r"[\W_]+", "", reason or "")


def parse_analysis(raw, transcript, existing_phrases, existing_reasons=(), existing_topics=()):
    """분석 응답 → [(phrase, reason, topic)]. 형식이 틀리면 None.

    phrase가 녹취(지금까지 + 이번에 판단한 것)에 실제로 없거나(지어낸 인용 — 판단을 미룬 최근 말도 여기 들지 않는다),
    이미 올린 것과 같은 표현이면 버린다. 표현은 달라도 사유나 안건(topic)이 이미 올린 것과 같으면 같은 문제를 다른
    표현으로 다시 올린 것이라 버린다(예: "최대한 빨리"와 "대충이라도 나오면"에 똑같이 "자료조사 기한").
    """
    data = _json_object(raw) or _flags_only(raw)
    if data is None:
        return None
    squashed_new = decision_checks.squash_transcript(transcript)
    seen = {decision_checks.squash(phrase) for phrase in existing_phrases}
    seen_reasons = {_reason_key(reason) for reason in existing_reasons}
    seen_topics = {_reason_key(topic) for topic in existing_topics if topic}
    flags = []
    for row in data.get("flags") or []:
        if not isinstance(row, dict):
            continue
        phrase = " ".join(str(row.get("phrase", "")).split())[:300]
        reason = " ".join(str(row.get("reason", "")).split())
        topic = " ".join(str(row.get("topic", "")).split())[:60]
        key, reason_key, topic_key = decision_checks.squash(phrase), _reason_key(reason), _reason_key(topic)
        if (not phrase or not reason or key in seen or reason_key in seen_reasons or (topic_key and topic_key in seen_topics)
                or not decision_checks.quote_in_source(phrase, squashed_new)):
            continue
        seen.add(key)
        seen_reasons.add(reason_key)
        if topic_key:
            seen_topics.add(topic_key)
        flags.append((phrase, reason, topic))
        if len(flags) >= MAX_FLAGS_PER_ANALYSIS:
            break
    return flags


def _same_spot(left, right):
    left, right = decision_checks.squash(left), decision_checks.squash(right)
    return bool(left and right) and (left in right or right in left)


def _rule_flags(chunks, analyzed_seq, until_seq, flags):
    """이번에 판단할 말에서 규칙이 찾은 애매한 점(live_rules — 일정 충돌, 미룸). 이미 올렸거나 진행자가 지운 것과 같은
    자리면 뺀다. 일정 충돌의 덩어리는 판단을 미룬 뒤의 말까지 이어 본다("그럼 목요일 7시에" / "다음 회의 잡을까요?")."""
    spoken = [c for c in chunks if c.transcript]
    new = [i for i, c in enumerate(spoken) if analyzed_seq < c.seq <= until_seq]
    if not new:
        return []
    lines = [c.transcript for c in spoken]
    found = live_rules.find_conflicts(lines, new[0], new[-1]) + live_rules.find_deferrals(lines, new[0], new[-1])
    kept, taken = [], [(f.phrase, f.reason) for f in flags]
    for phrase, reason, topic in found:
        if any(_duplicate(phrase, reason, other, other_reason, rule=live_rules.is_rule_reason(other_reason))
               for other, other_reason in taken):
            continue
        kept.append((phrase, reason, topic))
        taken.append((phrase, reason))
    return kept


def _duplicate(phrase, reason, other, other_reason, rule):
    """같은 애매한 점인가. 표현이나 사유가 같으면 같다. 상대가 규칙이 올린 것이면 표현이 서로를 품거나("이따 단톡에서
    정해요"와 "단톡에서 정해요") 같은 요일·시각을 가리켜도("다음 주 목요일 4시"와 "목요일 오후 4시") 같다고 본다."""
    if decision_checks.squash(phrase) == decision_checks.squash(other) or _reason_key(reason) == _reason_key(other_reason):
        return True
    return rule and (_same_spot(phrase, other) or live_rules.same_conflict(phrase, other))


def _analyze(session_id, until_seq, context_seq=None):
    """analyzed_seq 다음부터 until_seq까지를 판단한다. until_seq 다음부터 context_seq까지(판단을 미룬 최근 말)는
    [바로 뒤에 이어진 말]로 보여 주기만 한다."""
    advanced = False
    try:
        session = LiveSession.objects.get(pk=session_id)
        chunks = list(session.chunks.filter(status=LiveChunk.STATUS_DONE, seq__lte=max(until_seq, context_seq or -1)))
        new_text = " ".join(
            c.transcript for c in chunks if session.analyzed_seq < c.seq <= until_seq and c.transcript).strip()
        before = " ".join(c.transcript for c in chunks if c.seq <= session.analyzed_seq and c.transcript).strip()
        after = " ".join(c.transcript for c in chunks if c.seq > until_seq and c.transcript).strip()
        limit = max(0, _max_context_chars() - len(new_text) - len(after))
        trimmed = len(before) > limit
        if trimmed:
            before = before[len(before) - limit:]
        if new_text:
            flags = list(session.flags.all())
            shown = [f for f in flags if not f.dismissed]
            new_lines = [c.transcript for c in chunks if session.analyzed_seq < c.seq <= until_seq and c.transcript]
            prompt = analyze_prompt(new_text, before,
                                    [f for f in shown if not f.resolved], [f for f in shown if f.resolved], trimmed,
                                    after_text=after, dismissed_flags=[f for f in flags if f.dismissed],
                                    hint_lines=decision_compiler.review_hint_lines(new_lines, LIVE_HINT_PHRASES))
            # 일정 충돌과 미룸은 규칙으로도 본다 — 가벼운 모델이 "목요일 7시 약속" 뒤의 "목요일 7시에 회의 잡을까요?"나
            # "장소는 이따 단톡에서 정해요"를 놓쳤다. AI가 실패해도 올린다(이 구간은 다음 분석이 다시 보지만, 같은 것은
            # 다시 올리지 않는다).
            conflicts = _rule_flags(chunks, session.analyzed_seq, until_seq, flags)
            parsed, model, raw = None, "", ""

            def on_try(name):
                with _lock:
                    _current_try[session_id] = (time.monotonic(), name)

            for _ in range(decision_compiler.PARSE_RETRIES + 1):
                try:
                    raw, model = services.generate_live(prompt, on_try=on_try)
                except OSError:
                    break
                except Exception:
                    # 응답 형식이 깨져 JSON을 못 읽는 등 예상하지 못한 오류 — 분석 스레드가 그냥 죽으면 실패로 남지 않아,
                    # 일시정지 중에는 1초마다 다시 불렀다. 실패로 처리해 판단 딜레이 시간만큼 쉬게 한다.
                    logger.exception("실시간 애매함 분석 중 예상하지 못한 오류(세션 %s)", session_id)
                    break
                if model.startswith("로컬"):
                    # 로컬 모델은 한 번에 수십 초라 다시 묻지 않는다 — 깨진 JSON도 flags·settled는 꺼내 읽는다.
                    if services._is_korean_or_english(raw):
                        parsed = parse_analysis(raw, before + " " + new_text, [f.phrase for f in flags],
                                                [f.reason for f in flags], [f.topic for f in flags])
                    break
                if not services._is_korean_or_english(raw):
                    continue
                parsed = parse_analysis(raw, before + " " + new_text, [f.phrase for f in flags],
                                        [f.reason for f in flags], [f.topic for f in flags])
                if parsed is not None:
                    break
            if parsed is None:
                # 연결이 끊겼거나 읽을 수 없는 응답 — 이 구간은 넘기지 않고 다음 분석 때 다시 읽는다.
                for phrase, reason, topic in conflicts:
                    LiveFlag.objects.create(session=session, phrase=phrase, reason=reason, topic=topic)
                logger.warning("실시간 애매함 분석 실패(세션 %s) — 일정 충돌 %d건만 올리고 다음 분석 때 다시 시도합니다",
                               session_id, len(conflicts))
                with _lock:
                    _last_failure[session_id] = time.monotonic()
                return
            # AI도 규칙이 이번에 또는 앞에서 올린 것과 같은 자리를 짚었으면 규칙의 것 하나만 남긴다.
            rule_phrases = [phrase for phrase, _, _ in conflicts] + [
                f.phrase for f in flags if live_rules.is_rule_reason(f.reason)]
            found = conflicts + [
                (phrase, reason, topic) for phrase, reason, topic in parsed
                if not any(_same_spot(phrase, other) or live_rules.same_conflict(phrase, other) for other in rule_phrases)
            ]
            # 앞에서 올린 애매한 점이 이번 말에서 정해졌으면 자동으로 '확인함'으로 바꾼다(진행자가 되돌릴 수 있다).
            settled = parse_settled(raw, new_text, shown)
            with transaction.atomic():
                for phrase, reason, topic in found:
                    LiveFlag.objects.create(session=session, phrase=phrase, reason=reason, topic=topic)
                now = timezone.now()
                for flag, quote in settled:
                    LiveFlag.objects.filter(pk=flag.pk, resolved=False, dismissed=False).update(
                        resolved=True, resolved_auto=True, resolved_at=now, resolved_by=None, settled_quote=quote)
                session.analyzed_seq = until_seq
                session.save(update_fields=["analyzed_seq"])
            advanced = True
            with _lock:
                _last_failure.pop(session_id, None)
            logger.info("실시간 애매함 분석(세션 %s, %s): 조각 ~%d 판단(뒤 %d조각 참고) / 새로 올림 %d건 / 자동 확인 %d건",
                        session_id, model, until_seq, max(0, (context_seq or until_seq) - until_seq), len(found),
                        len(settled))
        else:
            LiveSession.objects.filter(pk=session_id).update(analyzed_seq=until_seq)
            advanced = True
    finally:
        with _lock:
            _analyzing.discard(session_id)
            _current_try.pop(session_id, None)
    # 분석하는 동안 조각이 더 쌓였으면 이어서 본다(분석 중에는 새 분석을 시작하지 않는다 — 간격이 아직 안 됐으면 다음
    # 조각 때 본다). 실패했으면 곧바로 다시 부르지 않고 다음 조각 때 다시 본다 — AI 장애 때 호출을 계속 되풀이하지 않게.
    if advanced:
        maybe_analyze(LiveSession.objects.get(pk=session_id))


# ---------------------------------------------------------------------------
# 종료 — 조각을 합쳐 통합 전사로 넘긴다
# ---------------------------------------------------------------------------


def _finish(session_id):
    try:
        session = LiveSession.objects.select_related("meeting").get(pk=session_id)
        if session.recording_id:
            return
        paths = [chunk.file.path for chunk in session.chunks.order_by("seq") if chunk.file]
        if not paths:
            LiveSession.objects.filter(pk=session_id).update(error="녹음된 내용이 없습니다")
            return
        started = timezone.localtime(session.started_at)
        name = "실시간녹음_%s.wav" % started.strftime("%Y%m%d_%H%M")
        with tempfile.TemporaryDirectory(prefix="meetalign-live-") as out_dir:
            merged = os.path.join(out_dir, name)
            try:
                _, skipped = services.merge_audio_to_wav(paths, merged)
            except RuntimeError as error:
                LiveSession.objects.filter(pk=session_id).update(error=str(error)[:300])
                return
            with open(merged, "rb") as handle:
                # 등록만 한다 — 전사는 진행자가 회의 화면에서 '제출하고 회의록 만들기'를 누를 때 다른 업로드와 함께 한다.
                recording = Recording(meeting=session.meeting, status=Recording.STATUS_READY,
                                      speaker_count=session.speaker_count)
                recording.file.save(name, File(handle), save=True)
        warning = "읽지 못한 녹음 조각 %d개가 빠졌습니다." % skipped if skipped else ""
        LiveSession.objects.filter(pk=session_id).update(recording=recording, error=warning)
        if skipped:
            return  # 일부만 등록된 경우 복구에 쓸 원본 조각을 보관한다.
        # 합친 녹음이 생겼으니 조각 파일은 지운다(녹취 글은 남긴다 — 회의 중 화면 기록).
        for chunk in session.chunks.all():
            if chunk.file:
                chunk.file.delete(save=False)
        LiveChunk.objects.filter(session=session).update(file="")
    except Exception:
        logger.exception("live session finish failed (session %s)", session_id)
        LiveSession.objects.filter(pk=session_id).update(error="녹음을 합치지 못했습니다. 다시 시도해 주세요.")
    finally:
        with _lock:
            _finishing.discard(session_id)


def finish(session, wait=False):
    """세션을 끝내고 조각을 합쳐 녹음으로 등록한다. 이미 합치는 중이거나 합쳤으면 아무것도 하지 않는다.
    wait면 백그라운드로 넘기지 않고 이 자리에서 합친다(회의 종료 — views._end_live_sessions)."""
    with _lock:
        if session.id in _finishing:
            return
        _finishing.add(session.id)
    if wait:
        _finish(session.id)
    else:
        _run(_finish, session.id)


# 세션마다 마지막으로 녹음을 시작하거나 이어서 녹음한 때(datetime). '조각이 한동안 끊겼다'(views._pause_if_stalled)를 볼 때
# 마지막 조각 시각과 함께 본다 — 5분 쉬고 이어서 녹음하면 첫 조각이 오기 전에 마지막 조각(5분 전)만 보고 다시 일시정지로
# 돌렸다. 메모리에만 둔다 — 서버가 재시작되면 조각 시각과 세션 시작 시각만 본다.
_activity = {}


def touch(session_id):
    """녹음을 시작하거나 이어서 녹음했다."""
    with _lock:
        _activity[session_id] = timezone.now()


def last_touch(session_id):
    with _lock:
        return _activity.get(session_id)


# 세션마다 지금 녹음하는 화면(탭)의 표시. 같은 진행자가 탭 두 개에서 동시에 녹음하면 같은 소리가 두 번 올라와 녹취가
# 겹치고 녹음 시간이 부풀었다. 마지막으로 '이어서 녹음'을 누른 화면만 녹음 조각을 올릴 수 있다. 메모리에만 두므로
# 서버가 재시작되면 비고, 그때는 처음 조각을 올리는 화면이 다시 녹음하는 화면이 된다.
_recorders = {}


def claim_recorder(session_id, recorder):
    """이 화면이 녹음을 맡는다(이어서 녹음을 눌렀다). 다른 화면은 다음 조각을 올릴 때 멈춘다."""
    if recorder:
        with _lock:
            _recorders[session_id] = recorder


def is_recorder(session_id, recorder):
    """이 화면이 녹음을 맡은 화면인가. 맡은 화면이 없으면(서버 재시작) 이 화면이 맡는다. 표시가 없는 요청은 막지 않는다."""
    if not recorder:
        return True
    with _lock:
        current = _recorders.setdefault(session_id, recorder)
    return current == recorder


def resolved_flag_texts(meeting):
    """회의 중 진행자가 '처리됨'으로 체크한 애매한 점 — 결정 추출이 다시 문제 삼지 않게 넘긴다. 잘못 짚었다고 지운 것은
    처리한 것이 아니라서 넣지 않는다."""
    return [
        "%s%s — %s" % ("[%s] " % flag.topic if flag.topic else "", flag.phrase, flag.reason)
        for flag in _resolved(meeting).order_by("created_at", "id")
    ]


def resolved_warning_quotes(meeting):
    return list(_resolved(meeting).exclude(topic="").values_list("topic", "phrase"))


def _resolved(meeting):
    # 진행자가 직접 체크한 것만 — 자동 확인(resolved_auto)은 가벼운 모델의 판단이라 결정 확인이 다시 본다.
    return LiveFlag.objects.filter(session__meeting=meeting, resolved=True, dismissed=False, resolved_auto=False)
