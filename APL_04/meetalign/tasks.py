import logging
import threading

from django.conf import settings
from django.db import close_old_connections

from . import services
from .models import Meeting

logger = logging.getLogger(__name__)

_active = set()
_active_lock = threading.Lock()


def recover_stuck(meeting):
    with _active_lock:
        alive = meeting.id in _active
    if meeting.processing_status == Meeting.STATUS_PROCESSING and not alive:
        meeting.processing_status = Meeting.STATUS_FAILED
        meeting.save(update_fields=["processing_status"])


def process_recording(meeting_id):
    meeting = Meeting.objects.get(pk=meeting_id)
    try:
        transcript = services.transcribe(meeting.recording.path)
        if transcript is not None:
            meeting.transcript = transcript
            meeting.record = services.summarize(transcript)
        meeting.processing_status = Meeting.STATUS_DONE
    except Exception:
        logger.exception("recording processing failed for meeting %s", meeting_id)
        meeting.processing_status = Meeting.STATUS_FAILED
    meeting.save(update_fields=["transcript", "record", "processing_status"])


def start_processing(meeting_id):
    if not getattr(settings, "BACKGROUND_RECORDING", True):
        Meeting.objects.filter(pk=meeting_id).update(processing_status=Meeting.STATUS_PROCESSING)
        process_recording(meeting_id)
        return

    with _active_lock:
        _active.add(meeting_id)
    Meeting.objects.filter(pk=meeting_id).update(processing_status=Meeting.STATUS_PROCESSING)

    def run():
        try:
            process_recording(meeting_id)
        finally:
            with _active_lock:
                _active.discard(meeting_id)
            close_old_connections()

    threading.Thread(target=run, daemon=True).start()
