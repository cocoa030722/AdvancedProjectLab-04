import logging
import threading

from django.conf import settings
from django.db import close_old_connections

from . import services
from .models import Meeting, Recording

logger = logging.getLogger(__name__)

_active = set()
# 파일을 지운 뒤 남은 파일로 회의록을 다시 만드는 중인 회의 id(start_refresh).
_refreshing = set()
_active_lock = threading.Lock()
_meeting_locks = {}


def _meeting_lock(meeting_id):
    with _active_lock:
        return _meeting_locks.setdefault(meeting_id, threading.Lock())


def _transcript_sections(recordings):
    """회의 녹취 원문을 이루는 조각들. 텍스트 파일을 올린 순서대로 먼저 두고, 녹음 전사본은 그 뒤에 모은다.

    텍스트 파일(회의 메모·안건 정리)이 앞에 오면 요약·결정 추출이 그 글을 맥락으로 삼고 녹음을 읽는다.
    나중에 녹음을 더 올려도 텍스트 파일은 계속 맨 앞에 남는다.

    텍스트 파일은 녹음에서 나온 말이 아니라 사람이 적은 글이라, 머리말에 '텍스트 파일'을 붙여 녹음 전사본과
    구분한다. 조각이 하나뿐이면 머리말 없이 그 내용만 쓴다(녹음 하나만 올린 가장 흔한 경우).
    """
    transcribed = [r for r in recordings if r.transcript]
    audio = [r for r in transcribed if not r.is_text]
    text = [r for r in transcribed if r.is_text]
    if len(audio) + len(text) == 1:
        return [(audio or text)[0].transcript]
    return (["[텍스트 파일: %s]\n%s" % (r.filename, r.transcript) for r in text]
            + ["[%s]\n%s" % (r.filename, r.transcript) for r in audio])


def refresh_meeting(meeting):
    recordings = list(meeting.recordings.all())
    full_transcript = "\n\n".join(_transcript_sections(recordings))
    meeting.transcript = full_transcript
    meeting.record = services.summarize(full_transcript) if full_transcript else ""
    if any(r.status == Recording.STATUS_PROCESSING for r in recordings):
        status = Meeting.STATUS_PROCESSING
    elif any(r.status == Recording.STATUS_DONE for r in recordings):
        status = Meeting.STATUS_DONE
    elif recordings:
        status = Meeting.STATUS_FAILED
    else:
        status = ""
    meeting.processing_status = status
    meeting.save(update_fields=["transcript", "record", "processing_status"])


def process_recording(recording_id, refresh=True):
    recording = Recording.objects.select_related("meeting").get(pk=recording_id)
    meeting = recording.meeting
    with _meeting_lock(meeting.id):
        try:
            if recording.is_text:
                recording.transcript = services.read_text_transcript(recording.file.path)
            else:
                recording.transcript = services.transcribe(
                    recording.file.path, recording_id=recording.id, speaker_count=recording.speaker_count,
                ) or ""
            recording.status = Recording.STATUS_DONE
            recording.error_message = ""
        except Exception as exc:
            logger.exception("recording processing failed for recording %s", recording_id)
            recording.status = Recording.STATUS_FAILED
            recording.error_message = str(exc)[:300]
        recording.save(update_fields=["transcript", "status", "error_message"])
        # 여러 파일을 한 번에 제출하면 마지막 파일을 마친 뒤 한 번만 합치고 요약한다(start_processing).
        if refresh:
            refresh_meeting(meeting)


def recover_stuck(meeting):
    with _active_lock:
        alive = set(_active)
        refreshing = meeting.id in _refreshing
    stuck = [r for r in meeting.recordings.all() if r.status == Recording.STATUS_PROCESSING and r.id not in alive]
    if not stuck:
        # 파일을 지우고 회의록을 다시 만들던 중에 서버가 재시작되면 처리할 파일이 없는데도 '처리 중'에
        # 머문다. 그때는 남은 파일로 바로 다시 만든다.
        if (meeting.processing_status == Meeting.STATUS_PROCESSING and not refreshing
                and not any(r.status == Recording.STATUS_PROCESSING for r in meeting.recordings.all())):
            refresh_meeting(meeting)
        return
    for recording in stuck:
        recording.status = Recording.STATUS_FAILED
        recording.error_message = "서버가 재시작되어 처리가 중단되었습니다"
        recording.save(update_fields=["status", "error_message"])
    refresh_meeting(meeting)


def _process_batch(recording_ids):
    """제출된 파일들을 하나씩 처리한 뒤 회의록을 한 번만 만든다. 합치는 순서는 _transcript_sections가 정한다."""
    recordings = list(Recording.objects.filter(pk__in=recording_ids).select_related("meeting"))
    recordings.sort(key=lambda r: (r.is_text, recording_ids.index(r.id)))
    for recording in recordings:
        process_recording(recording.id, refresh=False)
    if recordings:
        meeting = recordings[0].meeting
        with _meeting_lock(meeting.id):
            refresh_meeting(meeting)


def start_processing(*recording_ids):
    """한 회의에 올린 파일들(하나 이상)을 처리한다. 한 번 제출한 파일들은 한 스레드에서 차례로 처리한다."""
    recording_ids = list(recording_ids)
    if not recording_ids:
        return
    if not getattr(settings, "BACKGROUND_RECORDING", True):
        _process_batch(recording_ids)
        return

    meeting_id = Recording.objects.values_list("meeting_id", flat=True).get(pk=recording_ids[0])
    with _active_lock:
        _active.update(recording_ids)
    Meeting.objects.filter(pk=meeting_id).update(processing_status=Meeting.STATUS_PROCESSING)

    def run():
        try:
            _process_batch(recording_ids)
        finally:
            with _active_lock:
                _active.difference_update(recording_ids)
            close_old_connections()

    threading.Thread(target=run, daemon=True).start()


def start_refresh(meeting_id):
    """파일을 지운 뒤 남은 파일로 녹취 원문과 회의록을 다시 만든다. 요약에 AI를 부르므로 백그라운드에서 한다."""
    if not getattr(settings, "BACKGROUND_RECORDING", True):
        refresh_meeting(Meeting.objects.get(pk=meeting_id))
        return
    with _active_lock:
        _refreshing.add(meeting_id)
    Meeting.objects.filter(pk=meeting_id).update(processing_status=Meeting.STATUS_PROCESSING)

    def run():
        try:
            with _meeting_lock(meeting_id):
                refresh_meeting(Meeting.objects.get(pk=meeting_id))
        except Meeting.DoesNotExist:
            pass
        except Exception:
            logger.exception("meeting refresh failed for meeting %s", meeting_id)
        finally:
            with _active_lock:
                _refreshing.discard(meeting_id)
            close_old_connections()

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# 결정 사항 추출 — 버튼을 누른 요청 안에서 기다리지 않고 백그라운드에서 돈다.
#
# 추출은 AI를 1~3번 차례로 부른다(추출 → 추천 → 재생성). 긴 회의면 몇 분이 걸려서 요청
# 안에서 기다리면 Cloudflare 터널의 100초 제한에 걸려 진행자 화면에 524 오류가 뜨고, 서버는 계속 돌아
# 결과를 저장하는데 진행자는 실패한 줄 알고 버튼을 또 눌렀다. 그래서 녹음 처리와 같은 방식으로 스레드에서
# 돌리고, 화면은 '추출 중'을 보여주며 스스로 새로 고친다(decision_review.html).
#
# 상태는 메모리에만 둔다(녹음 처리의 _active와 같다). 서버가 재시작되면 '추출 중' 표시가 사라지고
# 결정이 없으니 추출 버튼이 다시 보인다 — 진행자가 다시 누르면 된다.
# ---------------------------------------------------------------------------

_decision_jobs = {}  # meeting_id → {"running": bool, "error": str}


def decision_compile_state(meeting_id):
    """{"running": 추출 중인가, "error": 마지막 추출의 실패 사유(없으면 "")}."""
    with _active_lock:
        job = _decision_jobs.get(meeting_id) or {}
        return {"running": bool(job.get("running")), "error": job.get("error", "")}


def pop_decision_compile_error(meeting_id):
    """끝난 추출의 실패 사유를 한 번만 돌려준다 — 화면에 한 번 보여주면 지운다."""
    with _active_lock:
        job = _decision_jobs.get(meeting_id)
        if not job or job.get("running"):
            return ""
        _decision_jobs.pop(meeting_id, None)
        return job.get("error", "")


def _run_decision_compile(meeting_id):
    # 순환 import를 피하려고 여기서 부른다(decision_compiler가 services·models를 쓴다).
    from . import decision_compiler
    error = ""
    try:
        decision_compiler.compile_meeting(Meeting.objects.get(pk=meeting_id))
    except decision_compiler.DecisionCompileError as exc:
        error = str(exc)
    except Meeting.DoesNotExist:
        error = ""
    except Exception:
        logger.exception("decision compile failed for meeting %s", meeting_id)
        error = "결정 사항 추출 중 오류가 났습니다. 잠시 뒤 다시 시도해 주세요."
    with _active_lock:
        _decision_jobs[meeting_id] = {"running": False, "error": error}


def start_decision_compile(meeting_id):
    """결정 사항 추출을 시작한다. 이미 추출 중이면 아무것도 하지 않는다(버튼을 두 번 눌러도 한 번만 돈다).

    BACKGROUND_DECISIONS가 꺼져 있으면(테스트) 그 자리에서 끝까지 돈다. 어느 쪽이든 끝난 결과는
    decision_compile_state / pop_decision_compile_error로 읽는다.
    """
    with _active_lock:
        if (_decision_jobs.get(meeting_id) or {}).get("running"):
            return
        _decision_jobs[meeting_id] = {"running": True, "error": ""}
    if not getattr(settings, "BACKGROUND_DECISIONS", True):
        _run_decision_compile(meeting_id)
        return

    def run():
        try:
            _run_decision_compile(meeting_id)
        finally:
            close_old_connections()

    threading.Thread(target=run, daemon=True).start()
