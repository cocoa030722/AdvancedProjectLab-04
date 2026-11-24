import os

from django.contrib.auth import login, logout
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import transaction
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from . import decision_checks, decision_compiler, live, notifications, services, tasks
from .models import (
    ChatMessage, Decision, DecisionAgreement, Interpretation, LiveChunk, LiveFlag, LiveSession, Meeting,
    Notification, Question, Recording, Team, TEXT_TRANSCRIPT_EXTENSIONS,
)

ALLOWED_RECORDING_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".webm", ".mp4", ".flac",
    ".aac", ".wma", ".opus", ".amr", ".caf", ".avi", ".mov", ".mkv", ".3gp",
}
MAX_RECORDING_BYTES = 500 * 1024 * 1024
# 회의 원문 텍스트 파일(.txt·.md). 녹음과 같은 칸으로 받되, 회의 한 번 분량의 글은 이보다 훨씬 작다.
MAX_TEXT_TRANSCRIPT_BYTES = 5 * 1024 * 1024


def index(request):
    teams = request.user.teams.all() if request.user.is_authenticated else []
    return render(request, "meetalign/index.html", {"teams": teams})


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            login(request, form.save())
            return redirect("index")
    else:
        form = UserCreationForm()
    return render(request, "meetalign/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            next_url = request.GET.get("next", "")
            if url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
                return redirect(next_url)
            return redirect("index")
    else:
        form = AuthenticationForm(request)
    return render(request, "meetalign/login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("index")


@login_required
def team_join(request):
    error = ""
    if request.method == "POST":
        if request.POST.get("action") == "create":
            name = request.POST.get("name", "").strip()
            if not name or Team.objects.filter(name=name).exists():
                error = "팀 이름이 비었거나 이미 존재합니다."
            else:
                team = Team.objects.create(name=name, created_by=request.user)
                team.members.add(request.user)
                return redirect("meeting_list", team_id=team.id)
        else:
            code = request.POST.get("code", "").strip().upper()
            team = Team.objects.filter(invite_code=code).first()
            if team is None:
                error = "해당 초대 코드의 팀이 없습니다."
            else:
                team.members.add(request.user)
                return redirect("meeting_list", team_id=team.id)
    return render(request, "meetalign/team_join.html", {"error": error})


def _team(request, team_id):
    return get_object_or_404(Team, pk=team_id, members=request.user)


def _meeting(request, meeting_id):
    return get_object_or_404(Meeting, pk=meeting_id, team__members=request.user)


def _meeting_rows(meetings, user):
    """회의 목록에 뿌릴 (회의 + 결정 알림 배지) 묶음.

    배지는 '이 사람이 아직 읽지 않은 결정 알림'이 있는 회의에만 붙는다(공개 → "새 결정",
    공개 뒤 수정 → "수정됨"). 회의 개수와 무관하게 쿼리 1번으로 모은다(N+1 금지).
    """
    kinds = notifications.unread_kinds_by_meeting(user, meetings)
    rows = []
    for meeting in meetings:
        kind = kinds.get(meeting.id)
        rows.append({
            "meeting": meeting,
            "notice": kind,
            "notice_label": "수정됨" if kind == Notification.KIND_REVISED else "새 결정" if kind else "",
        })
    return rows


@login_required
def meeting_list(request, team_id):
    team = _team(request, team_id)
    if request.method == "POST" and request.POST.get("action") == "delete_team":
        if team.created_by_id != request.user.id:
            raise PermissionDenied
        team.delete()
        return redirect("index")
    meetings = list(team.meetings.all())
    return render(request, "meetalign/meeting_list.html", {
        "team": team, "meetings": meetings, "rows": _meeting_rows(meetings, request.user),
    })


def _parse_deadline(raw):
    if not raw:
        return None, None
    try:
        parsed = parse_date(raw)
    except ValueError:
        parsed = None
    if parsed is None:
        return None, "마감일 형식이 올바르지 않습니다."
    if parsed < timezone.localdate():
        return None, "마감일은 오늘 이후 날짜로 설정해주세요."
    return parsed, None


def _recording_error(recording):
    """녹음 칸에 올린 파일 하나를 검사한다. 문제가 없으면 None."""
    extension = os.path.splitext(recording.name)[1].lower()
    if extension in TEXT_TRANSCRIPT_EXTENSIONS:
        return "%s: 텍스트 파일은 아래 '텍스트 파일' 칸에 올려 주세요." % recording.name
    if extension not in ALLOWED_RECORDING_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_RECORDING_EXTENSIONS))
        return "%s: 지원하지 않는 녹음 형식입니다 (%s)." % (recording.name, allowed)
    if recording.size > MAX_RECORDING_BYTES:
        return "%s: 녹음 파일이 너무 큽니다 (500MB 이하만 가능)." % recording.name
    return None


# 녹음을 올릴 때 적는 말한 사람 수의 범위. 화자 분리 결과는 화자 A~Z로 적으므로 26명이 한계지만, 회의 녹음에
# 그만큼 말하는 일은 없어서 20명에서 자른다.
MAX_SPEAKER_COUNT = 20


def _speaker_counts(request, recordings):
    """녹음마다 적은 '말한 사람 수' → ([값 또는 None, ...] 녹음 순서대로, [오류 문구, ...]).

    업로드 화면은 녹음 줄마다 speaker_counts 칸을 두고 파일과 같은 순서로 보낸다. 비워 둔 칸, 또는 칸이 오지
    않은 녹음(스크립트가 돌지 않는 브라우저)은 None — 화자 분리가 스스로 추정한다.
    """
    raws = request.POST.getlist("speaker_counts")
    counts, errors = [], []
    for index, recording in enumerate(recordings):
        raw = raws[index].strip() if index < len(raws) else ""
        if not raw:
            counts.append(None)
        elif raw.isdigit() and 1 <= int(raw) <= MAX_SPEAKER_COUNT:
            counts.append(int(raw))
        else:
            counts.append(None)
            errors.append("%s: 말한 사람 수는 1~%d 사이 숫자로 적어 주세요(모르면 비워 두세요)."
                          % (recording.name, MAX_SPEAKER_COUNT))
    return counts, errors


def _text_file_error(text_file):
    """텍스트 파일 칸에 올린 파일 하나를 검사한다. 읽을 수 없거나 빈 파일은 처리 단계까지 보내지 않는다."""
    extension = os.path.splitext(text_file.name)[1].lower()
    if extension not in TEXT_TRANSCRIPT_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in TEXT_TRANSCRIPT_EXTENSIONS))
        return "%s: 텍스트 파일 칸에는 %s 파일만 올릴 수 있어요." % (text_file.name, allowed)
    if text_file.size > MAX_TEXT_TRANSCRIPT_BYTES:
        return "%s: 텍스트 파일이 너무 큽니다 (5MB 이하만 가능)." % text_file.name
    try:
        text = services.decode_text_transcript(text_file.read())
    except ValueError as error:
        return "%s: %s" % (text_file.name, error)
    finally:
        text_file.seek(0)
    if not text:
        return "%s: 텍스트 파일이 비어 있습니다." % text_file.name
    return None


def _duplicate_text_errors(meeting, text_files):
    """이미 이 회의에 올린 텍스트 파일, 또는 함께 올린 다른 파일과 내용이 같은 텍스트 파일을 막는다.

    같은 메모를 두 번 올리면 녹취 원문에 똑같은 글이 두 번 들어가고 요약·결정 추출도 그 내용을 두 번
    읽는다. 파일 이름이 아니라 내용으로 비교한다 — 같은 이름으로 다시 올리면 저장소가 이름 뒤에 임의의
    글자를 붙여 별개 파일처럼 보이기 때문이다.
    """
    seen = {r.transcript: r.filename for r in meeting.recordings.all() if r.is_text and r.transcript}
    errors = []
    for text_file in text_files:
        text = services.decode_text_transcript(text_file.read())
        text_file.seek(0)
        if text in seen:
            errors.append("%s: 이미 올린 텍스트 파일(%s)과 내용이 같아요." % (text_file.name, seen[text]))
        else:
            seen[text] = text_file.name
    return errors


def _team_members(team):
    return list(team.members.order_by("username"))


def _chosen_participants(request, team, host):
    """폼에서 고른 참석자. 팀 멤버가 아닌 id는 버리고, 진행자는 언제나 넣는다."""
    ids = {raw for raw in request.POST.getlist("participants") if raw.isdigit()}
    chosen = [member for member in team.members.all() if str(member.pk) in ids]
    if host is not None and all(member.pk != host.pk for member in chosen):
        chosen.append(host)
    return chosen


def _participant_ids(meeting):
    """참석자 칸에 체크해 둘 id. 참석자를 고르지 않은 회의는 팀 멤버 전원(알림 범위와 같은 기준)."""
    ids = set(meeting.participants.values_list("pk", flat=True))
    return ids or set(meeting.team.members.values_list("pk", flat=True))


def _participant_context(meeting, user):
    """회의 화면의 참석자 칸(진행자에게만 보인다) — 결정 알림을 받을 범위. 진행 중인 실시간 세션도 함께 넘긴다."""
    # 종료된 회의는 실시간 세션도 끝난 것으로 본다(회의 종료가 세션을 함께 끝낸다 — _end_live_sessions).
    live_active = None if meeting.ended else meeting.live_sessions.exclude(status=LiveSession.STATUS_ENDED).first()
    if live_active is not None:
        _pause_if_stalled(live_active)
    # 녹음한 조각이 하나라도 있어야 시작한 회의다. 세션을 열어 두기만 하고 녹음 전이면 '녹음 중'도 '일시정지'도 아니다.
    live_started = live_active is not None and live_active.chunks.exists()
    # 회의 화면 위의 실시간 회의 배너. 녹음을 시작한 세션만 띄운다. 진행자에게는 일시정지일 때만 — 녹음 중이면 진행자는
    # 실시간 화면에 있다. 팀원에게는 녹음 중일 때(들어와서 함께 보라고)와 일시정지일 때. 녹음 중인데 조각이 끊겼으면
    # (화면을 닫았는데 일시정지 신호가 못 갔다) 위에서 일시정지로 돌려 둔다.
    live_paused = live_started and live_active.status == LiveSession.STATUS_PAUSED
    is_host = meeting.host_id == user.id
    live_banner = live_started and (live_paused or not is_host)
    live_recordings = dict(meeting.live_sessions.exclude(recording=None).values_list("recording_id", "error"))
    context = {"live_active": live_active, "live_started": live_started, "live_paused": live_paused,
               "live_banner": live_banner, "host_id": meeting.host_id,
               # 회의 파일 목록에서 '실시간 녹음'으로 표시할 녹음(실시간 세션을 끝내며 조각을 합친 것).
               "live_recording_ids": set(live_recordings),
               "live_partial_recording_ids": {recording_id for recording_id, warning in live_recordings.items() if warning}}
    if meeting.host_id != user.id:
        return {"team_members": [], "participant_ids": set(), **context}
    return {
        "team_members": _team_members(meeting.team),
        "participant_ids": _participant_ids(meeting),
        **context,
    }


@login_required
def meeting_create(request, team_id):
    team = _team(request, team_id)
    error = None
    if request.method == "POST" and request.POST.get("title", "").strip():
        deadline, error = _parse_deadline(request.POST.get("deadline", "").strip())
        if error is None:
            meeting = Meeting.objects.create(team=team, title=request.POST["title"].strip(), deadline=deadline, host=request.user)
            meeting.participants.set(_chosen_participants(request, team, request.user))
            return redirect("meeting_detail", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_create.html", {
        "team": team, "error": error, "team_members": _team_members(team),
        # 처음에는 팀 전원을 체크해 둔다 — 빼는 쪽이 고르는 쪽보다 손이 덜 간다.
        "participant_ids": {member.pk for member in team.members.all()}, "host_id": request.user.id,
    })


@login_required
def meeting_detail(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    error = None
    if request.method == "POST":
        # 녹음 칸과 텍스트 파일 칸 — 화면에서 여러 개를 쌓아 두었다가 한 번에 제출한다.
        recordings = request.FILES.getlist("recording")
        text_files = request.FILES.getlist("text_file")
        uploads = recordings + text_files
        if request.POST.get("action") == "delete_meeting":
            if meeting.host_id != request.user.id:
                raise PermissionDenied
            team_id = meeting.team_id
            meeting.delete()
            return redirect("meeting_list", team_id=team_id)
        if request.POST.get("action") == "set_participants":
            if meeting.host_id != request.user.id:
                raise PermissionDenied
            meeting.participants.set(_chosen_participants(request, meeting.team, meeting.host))
            return redirect("meeting_detail", meeting_id=meeting.id)
        if uploads and meeting.ended:
            error = "종료된 회의에는 녹음을 올릴 수 없습니다."
        elif uploads:
            # 하나라도 잘못됐으면 아무것도 저장하지 않는다 — 일부만 올라가면 회의록이 빠진 채로 만들어진다.
            errors = ([e for e in map(_recording_error, recordings) if e]
                      + [e for e in map(_text_file_error, text_files) if e])
            if not errors:
                errors = _duplicate_text_errors(meeting, text_files)
            speaker_counts, speaker_errors = _speaker_counts(request, recordings)
            errors.extend(speaker_errors)
            error = "\n".join(errors) or None
        if error:
            return render(request, "meetalign/meeting_detail.html", {
                "meeting": meeting, "error": error, "stt_status": services.stt_status_text(),
                **_participant_context(meeting, request.user),
            })
        if request.POST.get("action") == "end":
            # 일시정지된 채 남은 실시간 세션도 끝내고 녹음을 합쳐 둔다 — 아래에서 함께 처리된다.
            _end_live_sessions(meeting)
        # 실시간 세션에서 등록만 해 둔 녹음은 '제출하고 회의록 만들기'나 '회의 종료'를 누를 때 함께 처리한다. 종료한
        # 회의에는 더 제출할 수 없어서, 종료할 때 처리하지 않으면 등록된 녹음이 그대로 남는다.
        registered = []
        if request.POST.get("action") in ("submit", "end"):
            registered = list(meeting.recordings.filter(status=Recording.STATUS_READY).values_list("id", flat=True))
            Recording.objects.filter(pk__in=registered).update(status=Recording.STATUS_PROCESSING)
        if registered and not uploads:
            tasks.start_processing(*registered)
        if uploads:
            # 녹음을 먼저 만들고 텍스트 파일을 나중에 만든다(처리 순서). 녹취 원문에서는 텍스트 파일이 앞에 온다(tasks).
            # 말한 사람 수는 녹음마다 따로 붙는다(텍스트 파일은 화자 분리를 하지 않는다).
            added = [
                Recording.objects.create(
                    meeting=meeting, file=upload, status=Recording.STATUS_PROCESSING,
                    speaker_count=speaker_counts[index] if index < len(recordings) else None,
                )
                for index, upload in enumerate(uploads)
            ]
            tasks.start_processing(*registered, *(r.id for r in added))
        if request.POST.get("action") == "end":
            Meeting.objects.filter(pk=meeting.pk).update(ended=True)
            return redirect("meeting_detail", meeting_id=meeting.id)
        return redirect("meeting_result", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_detail.html", {
        "meeting": meeting, "stt_status": services.stt_status_text(),
        # 결정 사항 추출 기능(실험): 지난 회의의 미결정 안건. 읽기만 하므로 위 POST 흐름은 건드리지 않는다.
        # '처리됨' 버튼은 별도 URL(decision_carryover_resolve)로 보낸다.
        "carryover_decisions": decision_compiler.carryover_decisions(meeting),
        # 이 회의에서 읽지 않은 결정 알림(공개·수정) — 화면 위 안내로 보여준다.
        "decision_notice": notifications.unread_for_meeting(request.user, meeting),
        "notice_revised": Notification.KIND_REVISED,
        **_participant_context(meeting, request.user),
    })


def _end_live_sessions(meeting):
    """회의를 끝낼 때 아직 끝나지 않은(녹음 중·일시정지) 실시간 세션을 끝내고 조각을 하나의 녹음으로 합친다. 합치기는
    이 요청 안에서 끝낸다 — 합친 녹음을 같은 제출 묶음에 넣어야 회의록이 한 번에 만들어진다."""
    for session in meeting.live_sessions.exclude(status=LiveSession.STATUS_ENDED):
        LiveSession.objects.filter(pk=session.pk).update(status=LiveSession.STATUS_ENDED, ended_at=timezone.now())
        live.finish(session, wait=True)


@login_required
def meeting_result(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    tasks.recover_stuck(meeting)
    return render(request, "meetalign/meeting_result.html", {"meeting": meeting})


@login_required
def recording_file(request, meeting_id, recording_id):
    meeting = _meeting(request, meeting_id)
    recording = get_object_or_404(Recording, pk=recording_id, meeting=meeting)
    return FileResponse(recording.file.open("rb"), filename=recording.filename)


@login_required
def recording_retry(request, meeting_id, recording_id):
    meeting = _meeting(request, meeting_id)
    recording = get_object_or_404(Recording, pk=recording_id, meeting=meeting, status=Recording.STATUS_FAILED)
    if request.method == "POST":
        Recording.objects.filter(pk=recording.pk).update(status=Recording.STATUS_PROCESSING, error_message="")
        tasks.start_processing(recording.id)
    return redirect("meeting_result", meeting_id=meeting.id)


@login_required
@require_POST
def recording_speakers(request, meeting_id, recording_id):
    """등록한 녹음의 말한 사람 수를 고친다. 아직 전사하지 않은 녹음(등록됨)과 다시 시도할 녹음(실패)만 — 화자 분리는 전사할
    때 이 수를 쓰므로, 이미 전사가 끝난 녹음은 바꿔도 소용없다. 비우면 화자 분리가 사람 수를 스스로 짐작한다.
    실시간 녹음에서 합친 녹음은 녹음을 끝낼 때 적은 수가 미리 들어 있다."""
    meeting = _meeting(request, meeting_id)
    if meeting.host_id != request.user.id:
        raise PermissionDenied
    recording = get_object_or_404(
        Recording, pk=recording_id, meeting=meeting, status__in=[Recording.STATUS_READY, Recording.STATUS_FAILED],
    )
    count, errors = _speaker_counts_from(request.POST.get("speaker_count", ""))
    if meeting.ended or recording.is_text:
        errors = ["이 파일의 말한 사람 수는 바꿀 수 없습니다."]
    if errors:
        return render(request, "meetalign/meeting_detail.html", {
            "meeting": meeting, "error": errors[0], "stt_status": services.stt_status_text(),
            **_participant_context(meeting, request.user),
        })
    Recording.objects.filter(pk=recording.pk).update(speaker_count=count)
    return redirect("meeting_detail", meeting_id=meeting.id)


@login_required
@require_POST
def recording_delete(request, meeting_id, recording_id):
    """업로드된 녹음·텍스트 파일 하나를 지우고, 남은 파일로 녹취 원문과 회의록을 다시 만든다.

    진행자만 지울 수 있다(팀원은 올리기·다시 시도까지). 처리 중인 파일은 처리 스레드가 아직 그 행을
    쓰고 있으므로 끝난 뒤에 지우게 하고, 종료된 회의는 업로드와 마찬가지로 손대지 않는다.
    """
    meeting = _meeting(request, meeting_id)
    if meeting.host_id != request.user.id:
        raise PermissionDenied
    recording = get_object_or_404(Recording, pk=recording_id, meeting=meeting)
    if meeting.ended or recording.status == Recording.STATUS_PROCESSING:
        return redirect("meeting_detail", meeting_id=meeting.id)
    recording.file.delete(save=False)
    recording.delete()
    tasks.start_refresh(meeting.id)
    return redirect("meeting_detail", meeting_id=meeting.id)


# ---------------------------------------------------------------------------
# 실시간 회의 세션(meetalign.live). 녹음·일시정지·종료·애매한 점 처리는 진행자만, 보기는 팀원 누구나.
# ---------------------------------------------------------------------------

# 실시간 조각 하나의 크기 상한. 15초 브라우저 녹음(webm/opus)은 수백 KB라 넉넉하다.
MAX_LIVE_CHUNK_BYTES = 25 * 1024 * 1024


def _live_session(request, meeting_id, session_id):
    meeting = _meeting(request, meeting_id)
    session = get_object_or_404(LiveSession, pk=session_id, meeting=meeting)
    return meeting, session


def _live_host_only(meeting, user):
    if meeting.host_id != user.id:
        raise PermissionDenied


@login_required
@require_POST
def live_start(request, meeting_id):
    """진행자가 '실시간 녹음'을 누르면 세션을 연다. 이미 진행 중인 세션이 있으면 그쪽으로 보낸다."""
    meeting = _meeting(request, meeting_id)
    _live_host_only(meeting, request.user)
    if meeting.ended:
        return redirect("meeting_detail", meeting_id=meeting.id)
    session = meeting.live_sessions.exclude(status=LiveSession.STATUS_ENDED).first()
    if session is None:
        session = LiveSession.objects.create(meeting=meeting, started_by=request.user)
    return redirect("live_session", meeting_id=meeting.id, session_id=session.id)


@login_required
def live_session(request, meeting_id, session_id):
    meeting, session = _live_session(request, meeting_id, session_id)
    last_chunk = session.chunks.order_by("-seq").first()
    last = last_chunk.seq if last_chunk else None
    recording_elsewhere = False
    if session.status == LiveSession.STATUS_RECORDING and last_chunk is not None:
        recent = (timezone.now() - last_chunk.created_at).total_seconds() <= live.chunk_seconds() * 2 + 5
        if recent:
            # 다른 화면(탭)이 지금도 조각을 올리고 있다 — 그 녹음을 멈추지 않는다.
            recording_elsewhere = True
        else:
            # 진행자가 녹음 중에 화면을 떠났다. 녹음은 화면이 열려 있을 때만 되므로 일시정지로 돌려 둔다.
            LiveSession.objects.filter(pk=session.pk).update(status=LiveSession.STATUS_PAUSED)
            session.status = LiveSession.STATUS_PAUSED
    return render(request, "meetalign/live.html", {
        "meeting": meeting, "session": session, "is_host": meeting.host_id == request.user.id,
        "started": last_chunk is not None,
        "chunk_seconds": live.chunk_seconds(), "max_speakers": MAX_SPEAKER_COUNT,
        "interval": live.analyze_interval(session), "interval_min": live.INTERVAL_MIN_SECONDS,
        "interval_max": live.INTERVAL_MAX_SECONDS, "interval_step": live.INTERVAL_STEP_SECONDS,
        # 다시 들어왔을 때 이어 붙일 순번. 0부터 다시 매기면 이미 있는 순번과 겹쳐 새 녹음이 버려진다.
        "next_seq": 0 if last is None else last + 1,
        "recorded_ms": _recorded_ms(session),
        "recording_elsewhere": recording_elsewhere,
    })


def _recorded_ms(session):
    return sum(session.chunks.values_list("duration_ms", flat=True))


# 녹음 중인데 이 시간(조각 길이의 배수 + 여유) 동안 새 조각이 없으면 진행자 화면이 닫혔거나 녹음이 끊긴 것이다.
LIVE_STALL_FACTOR = 3
LIVE_STALL_GRACE_SECONDS = 15


def _lines_until(chunks):
    """0번부터 빈틈없이 전사가 끝난(또는 실패한) 마지막 조각 순번. 화면은 여기까지만 받아 쌓는다 — 뒤 조각이 먼저 끝나도
    앞 조각이 아직 전사 중이면 기다려야, 화면이 받은 데까지의 표시(after)가 앞 조각을 건너뛰지 않는다. 끝내 오지 않은
    순번(올리기를 포기한 조각)은 잠시 뒤 건너뛴다(live.contiguous_until)."""
    return live.contiguous_until(sorted(chunks, key=lambda c: c.seq), -1)


def _pause_if_stalled(session, chunks=None):
    """녹음 중인데 조각이 한동안 끊겼으면 일시정지로 돌린다 — 진행자 화면이 닫혔는데 일시정지 신호가 오지 못한 경우다.
    마지막 조각과 마지막으로 이어서 녹음한 때(live.touch) 가운데 늦은 것부터 센다 — 오래 쉬고 이어서 녹음하면 첫 조각이 오기
    전에 다시 일시정지로 돌렸다. 녹음한 조각이 없으면(녹음 시작 전) 건드리지 않는다 — 시작도 안 한 회의를 일시정지로 보였다."""
    if session.status != LiveSession.STATUS_RECORDING:
        return
    if chunks is None:
        last_chunk = session.chunks.order_by("-created_at").values_list("created_at", flat=True).first()
    else:
        last_chunk = max((c.created_at for c in chunks), default=None)
    if last_chunk is None:
        return
    last_activity = max(t for t in (last_chunk, live.last_touch(session.id)) if t is not None)
    stall_after = live.chunk_seconds() * LIVE_STALL_FACTOR + LIVE_STALL_GRACE_SECONDS
    if (timezone.now() - last_activity).total_seconds() > stall_after:
        LiveSession.objects.filter(pk=session.pk).update(status=LiveSession.STATUS_PAUSED)
        session.status = LiveSession.STATUS_PAUSED


def _live_state(session, after=-1):
    """after는 화면이 이미 받아 쌓은 마지막 조각 순번이다 — 녹취 줄은 그 뒤 것만 보낸다(긴 회의에서 1초마다 녹취 전체를
    다시 보내지 않게). 처음 열거나 다시 들어온 화면은 -1을 보내 처음부터 받는다."""
    chunks = list(session.chunks.all())
    flags = list(session.flags.filter(dismissed=False))
    _pause_if_stalled(session, chunks)
    lines_until = _lines_until(chunks)
    finishing = live.is_finishing(session.id)
    return {
        "status": session.status,
        # 녹음한 조각이 있는가 — 없으면 화면은 '녹음 중'·'일시정지' 대신 '녹음 시작 전'을 보인다.
        "started": bool(chunks),
        "recorded_ms": sum(c.duration_ms for c in chunks),
        # 회의 기록 칸에 대화처럼 쌓이는 녹취 줄(조각 하나 = 한 줄). DB에 남아 있어 화면을 떠났다 돌아와도 다시 보인다.
        "lines": [
            {"seq": c.seq, "text": c.transcript, "at": timezone.localtime(c.created_at).strftime("%H:%M")}
            for c in sorted(chunks, key=lambda c: c.seq)
            if after < c.seq <= lines_until and c.status == LiveChunk.STATUS_DONE and c.transcript
        ],
        "lines_until": lines_until,
        "pending": sum(1 for c in chunks if c.status == LiveChunk.STATUS_PENDING),
        "failed": sum(1 for c in chunks if c.status == LiveChunk.STATUS_FAILED),
        "next_seq": max((c.seq for c in chunks), default=-1) + 1,
        "flags": [
            {"id": f.id, "topic": f.topic, "phrase": f.phrase, "reason": f.reason, "resolved": f.resolved,
             "auto": f.resolved and f.resolved_auto, "settled_quote": f.settled_quote,
             "at": timezone.localtime(f.created_at).strftime("%H:%M")}
            for f in flags
        ],
        # 애매함 분석 진행 — 화면이 "판단 중", "대기 중 · n초 남음" 같은 표시를 보여 준다(live.analysis_status).
        "analysis": {**live.analysis_status(session), "interval": live.analyze_interval(session)},
        "recording_ready": bool(session.recording_id),
        "recording_finishing": finishing,
        "can_retry_merge": not session.is_active and not session.recording_id and not finishing
                           and any(chunk.file for chunk in chunks),
        # 회의 종료로 세션이 끝났으면 합친 녹음은 이미 처리 중이다 — 화면이 '제출하라'고 안내하지 않게.
        "meeting_ended": Meeting.objects.filter(pk=session.meeting_id, ended=True).exists(),
        "error": session.error,
    }


@login_required
def live_state(request, meeting_id, session_id):
    """실시간 화면이 몇 초마다 묻는 현재 상태(JSON). 멈춘 조각이 있으면 다시 전사를 건다."""
    _, session = _live_session(request, meeting_id, session_id)
    if session.is_active:
        live.recover(session)
        # 분석은 조각 전사가 끝날 때 시작되는데, 조용하면 다음 조각이 올 때까지 남은 시간이 0인 채로 멈춰 보였다. 화면이
        # 1초마다 묻는 이때도 확인해서, 기다리는 시간이 다 되면 바로 시작한다(간격·잠금은 maybe_analyze가 지킨다).
        # 일시정지 중에도 본다 — 일시정지를 누른 뒤에 전사가 끝난 마지막 조각을 바로 판단하게(예전에는 이어서 녹음할
        # 때까지 '곧 판단해요'에 멈춰 있었다).
        live.maybe_analyze(session)
    raw_after = request.GET.get("after", "")
    after = int(raw_after) if raw_after.lstrip("-").isdigit() else -1
    return JsonResponse(_live_state(session, after=after))


@login_required
@require_POST
def live_chunk(request, meeting_id, session_id):
    """브라우저가 올리는 녹음 조각 하나. 같은 순번이 두 번 오면(재전송) 처음 것만 쓴다."""
    meeting, session = _live_session(request, meeting_id, session_id)
    _live_host_only(meeting, request.user)
    if not session.is_active:
        return JsonResponse({"error": "이미 끝난 세션입니다."}, status=409)
    upload = request.FILES.get("chunk")
    raw_seq = request.POST.get("seq", "")
    if upload is None or not raw_seq.isdigit():
        return HttpResponseBadRequest("chunk와 seq가 필요합니다.")
    if upload.size > MAX_LIVE_CHUNK_BYTES:
        return JsonResponse({"error": "녹음 조각이 너무 큽니다."}, status=413)
    if not live.is_recorder(session.id, request.POST.get("recorder", "")):
        # 다른 화면(탭)이 녹음을 맡았다. 이 화면은 녹음을 멈춘다(live.html).
        return JsonResponse({"error": "other_recorder"}, status=409)
    raw_duration = request.POST.get("duration_ms", "")
    duration_ms = min(int(raw_duration), 10 * 60 * 1000) if raw_duration.isdigit() else 0
    chunk, created = LiveChunk.objects.get_or_create(
        session=session, seq=int(raw_seq), defaults={"file": upload, "duration_ms": duration_ms},
    )
    if created:
        # 브라우저가 조각 동안 소리 크기를 재서 거의 조용했으면 silent=1을 붙인다 — 전사하지 않는다.
        live.transcribe_chunk(chunk, silent=request.POST.get("silent") == "1")
    return JsonResponse({"ok": True, "seq": chunk.seq})


@login_required
@require_POST
def live_control(request, meeting_id, session_id):
    """일시정지·이어서·종료. 종료하면 조각을 합쳐 녹음으로 등록하고(전사는 회의 화면에서 제출할 때) 끝낸다."""
    meeting, session = _live_session(request, meeting_id, session_id)
    _live_host_only(meeting, request.user)
    action = request.POST.get("action")
    if action == "retry_merge":
        if session.is_active or session.recording_id or not session.chunks.exclude(file="").exists():
            return JsonResponse({"error": "다시 합칠 녹음이 없습니다."}, status=409)
        live.finish(session)
        return JsonResponse({"status": session.status})
    if not session.is_active:
        return JsonResponse({"status": session.status})
    recorder = request.POST.get("recorder", "")
    if action == "pause":
        if not live.is_recorder(session.id, recorder):
            # 이미 다른 화면이 녹음을 맡았다 — 떠나는 옛 화면의 일시정지 신호가 지금 녹음을 멈추지 않게 한다.
            return JsonResponse({"status": session.status})
        LiveSession.objects.filter(pk=session.pk).update(status=LiveSession.STATUS_PAUSED)
        # 멈춘 김에 그때까지 쌓인 녹취를 바로 본다 — 쉬는 동안 애매한 점을 정리할 수 있게.
        live.maybe_analyze(session, force=True)
    elif action == "resume":
        live.claim_recorder(session.id, recorder)
        live.touch(session.id)  # 첫 조각이 오기 전에 '끊겼다'고 보지 않게
        LiveSession.objects.filter(pk=session.pk).update(status=LiveSession.STATUS_RECORDING)
    elif action == "interval":
        # 판단 딜레이 시간(분석 간격) — '애매한 점' 아래 슬라이더. 바로 다음 분석부터 적용된다.
        raw = request.POST.get("seconds", "")
        if not raw.isdigit():
            return HttpResponseBadRequest("seconds가 필요합니다.")
        LiveSession.objects.filter(pk=session.pk).update(analyze_interval=live.clamp_interval(int(raw)))
        session.refresh_from_db()
        return JsonResponse({"status": session.status, "interval": live.analyze_interval(session)})
    elif action == "end":
        counts, errors = _speaker_counts_from(request.POST.get("speaker_count", ""))
        if errors:
            return JsonResponse({"error": errors[0]}, status=400)
        LiveSession.objects.filter(pk=session.pk).update(
            status=LiveSession.STATUS_ENDED, ended_at=timezone.now(), speaker_count=counts,
        )
        session.refresh_from_db()
        live.finish(session)
    else:
        return HttpResponseBadRequest("알 수 없는 동작입니다.")
    session.refresh_from_db()
    return JsonResponse({"status": session.status})


def _speaker_counts_from(raw):
    """종료할 때 적은 말한 사람 수 하나 → (값 또는 None, [오류])."""
    raw = (raw or "").strip()
    if not raw:
        return None, []
    if raw.isdigit() and 1 <= int(raw) <= MAX_SPEAKER_COUNT:
        return int(raw), []
    return None, ["말한 사람 수는 1~%d 사이 숫자로 적어 주세요(모르면 비워 두세요)." % MAX_SPEAKER_COUNT]


@login_required
@require_POST
def live_flag(request, meeting_id, session_id, flag_id):
    """진행자가 애매한 점을 '처리됨'으로 체크하거나 되돌린다. 처리한 것은 결정 추출이 다시 문제 삼지 않는다.
    dismissed=1이면 잘못 짚은 것으로 지운다(화면의 ×) — 모두의 화면에서 사라지고, 다음 분석은 같은 것을 다시 올리지 않는다."""
    meeting, session = _live_session(request, meeting_id, session_id)
    _live_host_only(meeting, request.user)
    flag = get_object_or_404(LiveFlag, pk=flag_id, session=session)
    if "dismissed" in request.POST:
        flag.dismissed = request.POST.get("dismissed") == "1"
        flag.save(update_fields=["dismissed"])
        return JsonResponse({"id": flag.id, "dismissed": flag.dismissed})
    resolved = request.POST.get("resolved") == "1"
    flag.resolved = resolved
    flag.resolved_by = request.user if resolved else None
    flag.resolved_at = timezone.now() if resolved else None
    # 진행자가 직접 고르면 자동 확인이 아니다. 자동 확인을 되돌린 것은 settled_quote가 남아 다시 자동 확인하지 않는다.
    flag.resolved_auto = False
    flag.save(update_fields=["resolved", "resolved_by", "resolved_at", "resolved_auto"])
    return JsonResponse({"id": flag.id, "resolved": flag.resolved})


CHAT_HISTORY_TURNS = 3


@login_required
def chat(request, meeting_id):
    """대화창 — LLM과 회의 내용을 묻고 답하는 화면 하나뿐이다. 주최자에게 보내는 익명 질문은 익명 대화창(inbox)으로 옮겼다."""
    meeting = _meeting(request, meeting_id)
    # 예전 대화창의 '주최자에게 익명 질문' 탭으로 오던 요청(북마크, 열어 둔 옛 화면)은 익명 대화창으로 보낸다.
    # 보내던 질문은 LLM에 넘기지 않는다 — 쓴 사람은 이름 없이 주최자에게 보낸다고 생각하고 쓴 글이다.
    if (request.POST.get("mode") or request.GET.get("mode")) == "anon":
        if request.method == "POST":
            return _post_anonymous_question(request, meeting)
        return redirect("inbox", meeting_id=meeting.id)
    if request.method == "POST" and request.POST.get("text", "").strip():
        text = request.POST["text"].strip()
        history = list(
            ChatMessage.objects.filter(meeting=meeting, user=request.user)
            .order_by("-created_at")[: CHAT_HISTORY_TURNS * 2]
            .values_list("role", "text")
        )[::-1]
        ChatMessage.objects.create(meeting=meeting, user=request.user, role=ChatMessage.ROLE_USER, text=text)
        reply = services.chat_reply(meeting, text, history=history)
        ChatMessage.objects.create(meeting=meeting, user=request.user, role=ChatMessage.ROLE_ASSISTANT, text=reply)
        return redirect("chat", meeting_id=meeting.id)
    messages = ChatMessage.objects.filter(meeting=meeting, user=request.user)
    return render(request, "meetalign/chat.html", {"meeting": meeting, "messages": messages})


def _require_host(meeting, user):
    # 주최자가 없는(이 기능 도입 전에 만든) 회의는 팀원 누구나 허용한다.
    if meeting.host_id is not None and meeting.host_id != user.id:
        raise PermissionDenied


def _render_inbox(request, meeting, error=None, text=""):
    # 질문과 답변은 팀원 모두에게 공개하고, 작성자는 누구에게도(주최자 포함) 보여주지 않는다.
    # 답변은 주최자만 단다(answers 뷰). is_host는 답변하기 버튼 노출에만 쓴다.
    # 결정 카드에서 온 질문은 어떤 결정에 대한 질문인지 함께 보여주므로 결정도 같이 읽는다.
    questions = meeting.questions.select_related("decision").order_by("-created_at", "-id")
    return render(request, "meetalign/inbox.html", {
        "meeting": meeting, "questions": questions, "is_host": meeting.host_id == request.user.id,
        "error": error, "text": text, "sent": request.GET.get("sent") == "1",
    })


def _post_anonymous_question(request, meeting):
    """익명 대화창에서 쓴 질문을 저장한다. 작성자(user)는 '내 질문' 표시에만 쓰고 누구에게도 보이지 않는다."""
    text = request.POST.get("text", "").strip()
    if not text:
        return _render_inbox(request, meeting, error="질문을 입력해 주세요.")
    if meeting.ended:
        return _render_inbox(request, meeting, error="종료된 회의에는 익명 질문을 보낼 수 없습니다.", text=text)
    Question.objects.create(meeting=meeting, text=text, user=request.user)
    return redirect("%s?sent=1" % reverse("inbox", args=[meeting.id]))


@login_required
def inbox(request, meeting_id):
    """익명 대화창 — 주최자에게 이름 없이 질문을 보내고, 팀원 모두의 질문과 주최자 답변을 한곳에서 본다."""
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        if "question_id" in request.POST:
            return _save_answer(request, meeting)
        return _post_anonymous_question(request, meeting)
    return _render_inbox(request, meeting)


def _save_answer(request, meeting):
    """주최자가 익명 대화창에서 질문 아래 바로 답을 단다. 답을 비우고 저장하면 '아직 답변 전'으로 돌아간다."""
    _require_host(meeting, request.user)
    question = get_object_or_404(Question, pk=request.POST.get("question_id"), meeting=meeting)
    question.answer = request.POST.get("answer", "").strip()
    question.save(update_fields=["answer"])
    return redirect("%s#q%d" % (reverse("inbox", args=[meeting.id]), question.id))


@login_required
def summary(request, meeting_id):
    return render(request, "meetalign/summary.html", {"meeting": _meeting(request, meeting_id)})


@login_required
def answers(request, meeting_id):
    """예전 '질문에 답변하기' 화면 주소. 답변은 이제 익명 대화창 안에서 단다 — 옛 링크와 열어 둔 옛 화면을 위해 남겨 둔다."""
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        return _save_answer(request, meeting)
    return redirect("inbox", meeting_id=meeting.id)


@login_required
def record_file(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if not meeting.record:
        raise Http404
    response = HttpResponse(meeting.record, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="meeting_record.txt"'
    return response


# ---------------------------------------------------------------------------
# 결정 사항 추출 기능. 회의 원문에서 뽑은 결정 사항을 진행자가 확정해 공개하고, 팀원이
# 항목마다 확인 여부를 남기는 흐름이다.
# ---------------------------------------------------------------------------


def _decision_meeting(request, meeting_id):
    """결정 사항 추출 기능을 쓸 수 있는 회의만 통과시킨다.

    팀 밖 사용자는 _meeting()이 404로 막고, 회의록이 아직 완성되지 않은 회의는 여기서 404다
    — 추출할 녹취 원문이 없는 회의에 링크를 노출하지 않기 위함(템플릿 쪽도 같은 조건).
    """
    meeting = _meeting(request, meeting_id)
    if meeting.processing_status != Meeting.STATUS_DONE:
        raise Http404
    return meeting


def _reset_decisions(meeting):
    """추출 결과를 전부 지운다. 공개된 뒤에는 거부한다.

    팀원이 이미 본 내용이 말없이 사라지면 안 되기 때문이다. 다시 하려면 먼저 '공개 취소'로
    draft로 내려야 한다.
    """
    if meeting.decisions.exclude(status=Decision.STATUS_DRAFT).exists():
        return "이미 공개된 항목이 있어 결정 사항을 지울 수 없습니다. '공개 취소'를 먼저 눌러 주세요."
    meeting.decisions.all().delete()
    return None


def _unpublish_decisions(meeting):
    """공개를 되돌려 draft로 내린다 — 공개 직후 "잘못 눌렀다"를 수습하는 통로.

    팀원 확인을 받지 않으므로 언제든 되돌릴 수 있다. 아직 읽지 않은 알림은 거둔다 — 눌러도 볼
    결정이 없기 때문이다.
    """
    published = meeting.decisions.exclude(status=Decision.STATUS_DRAFT)
    if not published.exists():
        return "공개된 항목이 없습니다."
    published.update(status=Decision.STATUS_DRAFT, revised=False)
    notifications.withdraw_unread(meeting)
    return None


def _planned_choice(request, decision):
    """POST에서 이 항목의 선택을 읽어 ((종류, 값) 또는 None, 실패 사유 또는 None)을 돌려준다.

    읽기만 하고 쓰지 않는다 — 한 제출 안의 모든 선택을 먼저 검증한 뒤 한꺼번에 적용하기
    위함이다. 그러지 않으면 뒤쪽 항목이 거부당했는데 앞쪽의 '이 결정 빼기'는 이미 커밋되어,
    진행자가 되돌릴 방법이 없다(다시 추출하면 다른 선택도 전부 사라진다).
    """
    raw = request.POST.get("choice_%d" % decision.id, "")
    if not raw:
        return None, None

    if raw in (decision_compiler.CHOICE_DROP, decision_compiler.CHOICE_UNDECIDED):
        return (raw, None), None

    if raw == decision_compiler.CHOICE_CUSTOM:
        text = request.POST.get("custom_text_%d" % decision.id, "").strip()
        if not text:
            return None, "'직접 쓰기'를 골랐으면 확정 문장을 한 줄 적어 주세요."
        return (decision_compiler.CHOICE_CUSTOM, text), None

    if raw.startswith(decision_compiler.CHOICE_CANDIDATE_PREFIX):
        candidate_id = raw[len(decision_compiler.CHOICE_CANDIDATE_PREFIX):]
        if not candidate_id.isdigit():
            return None, "확정 방법을 하나 골라 주세요."
        interpretation = get_object_or_404(Interpretation, pk=candidate_id, decision=decision)
        # 고른 시점의 문장을 복사해 둔다. 뒤에 후보 문장이 바뀌어도 확정 내용은 따라 바뀌지 않는다.
        return (decision_compiler.CHOICE_CANDIDATE_PREFIX, interpretation.text), None

    return None, "확정 방법을 하나 골라 주세요."


def _apply_planned_choice(decision, kind, value):
    """검증을 통과한 선택 하나를 적용한다. draft인 동안에만 쓴다(동시 요청 대비 조건부 쓰기)."""
    draft_only = Decision.objects.filter(pk=decision.pk, status=Decision.STATUS_DRAFT)
    if kind == decision_compiler.CHOICE_DROP:
        # 회의에 없던 결정을 지운다. Interpretation은 cascade로 함께 사라진다.
        draft_only.delete()
        return
    if kind == decision_compiler.CHOICE_UNDECIDED:
        # 미정 항목도 문장을 갖는다 — 팀원 화면과 이월 목록이 final_text 하나만 보기 때문이다.
        draft_only.update(
            final_text=decision_compiler.undecided_sentence(decision.what), marked_undecided=True)
        return
    draft_only.update(final_text=value, marked_undecided=False)


def _save_decision_edits(request, meeting, publishing=False):
    """진행자의 확정 선택을 저장한다. draft 항목만 대상이다.

    고를 수 있는 것은 [AI 추천 문장 | 직접 쓰기 | 다음 회의로 넘김 | 이 결정 빼기]뿐이다.
    라디오를 고르지 않은 항목은 이렇게 다룬다.
    - 그대로 공개 항목(경고 없음): '직접 쓰기' 칸에 적힌 문장을 확정 문장으로 쓴다. 칸에는 AI가 쓴
      문장이 미리 들어 있으므로, 진행자가 칸만 고치면 라디오를 누르지 않아도 고친 문장이 공개된다.
      칸을 비웠으면 원래 문장을 그대로 둔다.
    - 확인 필요 항목: 저장할 때는 그대로 두고, 공개할 때(publishing=True)는 '이 결정 빼기'로
      처리한다 — 모든 항목을 고르지 않아도 공개할 수 있게 하되, 진행자가 확정하지 않은 모호한
      결정이 팀원에게 나가지 않게 한다. 다만 이렇게 빼면 공개할 항목이 하나도 남지 않는 경우에는
      빼지 않고 사유를 돌려준다(되돌릴 수 없는 삭제로 회의 결정이 통째로 사라지지 않게).

    모든 선택을 먼저 검증한다. 하나라도 잘못되면 **되돌릴 수 없는 '이 결정 빼기'만 보류**하고
    나머지는 적용한다 — 진행자가 고른 것이 사라지지 않아야 하고(화면이 DB를 다시 읽어 그리므로
    적용하지 않으면 그 선택은 화면에서도 사라진다), 동시에 실패한 제출이 삭제를 커밋해 복구
    불가능한 상태를 만들어서도 안 된다. 실패 사유 문자열 또는 None을 돌려준다.
    """
    planned = []
    implicit_drops = []
    error = None
    drafts = []
    for decision in meeting.decisions.all():
        if decision.status != Decision.STATUS_DRAFT:
            # 공개된 항목은 선택을 받지 않는다. 다만 '이 결정 빼기'는 조용히 무시하지 않고
            # 사유를 돌려준다 — 진행자가 왜 안 됐는지 알아야 '확정 문장 수정'으로 갈 수 있다.
            if request.POST.get("choice_%d" % decision.id, "") == decision_compiler.CHOICE_DROP and error is None:
                error = "공개된 항목은 뺄 수 없습니다. 내용을 고치려면 '확정 문장 수정'을 쓰세요."
            continue
        drafts.append(decision)
        plan, failed = _planned_choice(request, decision)
        if failed and error is None:
            # 어느 항목인지 함께 알려 준다 — 항목이 많으면 사유만으로는 찾을 수 없다.
            error = "'%s': %s" % (decision.what, failed)
        elif plan:
            planned.append((decision, plan))
        elif not failed and not decision.marked_undecided:
            if not decision.needs_host:
                text = request.POST.get("custom_text_%d" % decision.id, "").strip()
                if text and text != decision.final_text:
                    planned.append((decision, (decision_compiler.CHOICE_CUSTOM, text)))
            elif publishing and decision.blocks_publish:
                implicit_drops.append((decision, (decision_compiler.CHOICE_DROP, None)))
    if implicit_drops and error is None:
        dropped = {decision.pk for decision, _ in implicit_drops} | {
            decision.pk for decision, plan in planned if plan[0] == decision_compiler.CHOICE_DROP
        }
        if all(decision.pk in dropped for decision in drafts):
            error = "고른 결정이 하나도 없어 공개하면 모든 결정이 빠집니다. 하나 이상 골라 주세요."
        else:
            planned.extend(implicit_drops)
    if error:
        # 삭제는 되돌릴 수 없으므로 제출이 깨끗할 때만 한다. 나머지 선택은 그대로 저장한다.
        planned = [
            (decision, plan) for decision, plan in planned
            if plan[0] != decision_compiler.CHOICE_DROP
        ]
    with transaction.atomic():
        for decision, (kind, value) in planned:
            _apply_planned_choice(decision, kind, value)
    return error


def _publish_decisions(meeting):
    """"확정 문장 공개" — draft 항목을 팀원에게 공개한다.

    고르지 않은 확인 필요 항목은 바로 앞의 _save_decision_edits(publishing=True)가 이미 뺐다.
    그래도 공개할 문장이 없는 항목이 남아 있으면 거부한다(마지막 그물).
    """
    drafts = list(meeting.decisions.filter(status=Decision.STATUS_DRAFT))
    if not drafts:
        if not meeting.decisions.exists():
            return "모든 결정을 뺐습니다. 다시 추출해 주세요."
        return "공개할 항목이 없습니다."
    blockers = decision_compiler.publish_blockers(drafts)
    if blockers:
        return "공개할 문장이 없는 항목이 %d개 있습니다. 추천 문장을 고르거나 직접 쓰거나 '다음 회의로 넘김'을 골라 주세요." % len(blockers)
    decision_compiler.publish_decisions(meeting)
    notifications.notify_decision_change(meeting, Notification.KIND_PUBLISHED)
    return None


def _revise_final_text(request, meeting):
    """공개된 항목의 확정 문장을 고친다. 팀원에게 수정 알림이 간다."""
    decision = get_object_or_404(
        Decision, pk=request.POST.get("decision_id"), meeting=meeting,
        status=Decision.STATUS_PUBLISHED,
    )
    text = request.POST.get("final_text", "").strip()
    if not text:
        return "확정 문장을 한 줄 적어 주세요."
    if text == decision.final_text:
        # 같은 문장이면 바뀐 것이 없으므로 알림도 보내지 않는다.
        return None
    decision_compiler.revise_final_text(decision, text)
    notifications.notify_decision_change(meeting, Notification.KIND_REVISED)
    return None


def _resolve_carryover(meeting, decision_id):
    """이월된 미결정 안건을 '처리됨'으로 닫는다. 이후 회의 화면에서 사라진다."""
    decision = get_object_or_404(
        Decision, pk=decision_id, meeting__team_id=meeting.team_id,
        status=Decision.STATUS_UNDECIDED, carryover_resolved_at__isnull=True,
    )
    Decision.objects.filter(pk=decision.pk).update(carryover_resolved_at=timezone.now())


@login_required
def decision_carryover_resolve(request, meeting_id, decision_id):
    """이월 안건 '처리됨' 전용 엔드포인트.

    회의 상세 화면(meeting_detail)의 기존 POST 흐름(녹음 업로드/종료/삭제)을 건드리지 않기
    위해 별도 URL로 뺐다. 지금 보고 있는 회의의 진행자만 누를 수 있다.
    """
    meeting = _meeting(request, meeting_id)
    _require_host(meeting, request.user)
    if request.method == "POST":
        _resolve_carryover(meeting, decision_id)
    return redirect("meeting_detail", meeting_id=meeting.id)



def _current_choice(decision, interpretations):
    """진행자 확정 라디오에서 지금 선택돼 있어야 하는 값과, '직접 작성' 입력칸에 넣을 문장.

    고른 시점에 후보 문장을 final_text로 복사해 두므로, 같은 문장을 가진 후보가 있으면 그
    후보가 선택된 것으로 본다. 어느 후보와도 다르면 직접 작성한 것이다.
    """
    if decision.marked_undecided:
        return decision_compiler.CHOICE_UNDECIDED, ""
    if not decision.has_warning:
        # 확인이 필요 없는 항목은 아무것도 선택돼 있지 않아야 한다('고치기'는 접혀 있다).
        # '직접 쓰기' 칸에는 AI가 쓴 문장을 미리 넣어 고치기 쉽게 한다.
        return "", decision.final_text
    if not decision.final_text:
        return "", ""
    for interpretation in interpretations:
        if interpretation.text == decision.final_text:
            return "%s%d" % (decision_compiler.CHOICE_CANDIDATE_PREFIX, interpretation.id), ""
    return decision_compiler.CHOICE_CUSTOM, decision.final_text


@login_required
def decision_review(request, meeting_id):
    """진행자 전용. 추출 → 결정 사항·경고 확인과 수정 → 항목별 확정 → "확정된 결정 공개"."""
    meeting = _decision_meeting(request, meeting_id)
    _require_host(meeting, request.user)
    error = None
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "compile":
            # 백그라운드에서 돈다(tasks.start_decision_compile). 이미 추출한 회의는 시작하지 않고 바로 알린다.
            if meeting.decisions.exists():
                error = "이미 결정 사항을 추출한 회의입니다. 다시 하려면 기존 결정을 먼저 지워 주세요."
            else:
                tasks.start_decision_compile(meeting.id)
                # 그 자리에서 끝난 경우(백그라운드를 끈 설정)에는 실패 사유를 바로 보여준다.
                error = tasks.pop_decision_compile_error(meeting.id) or None
        elif action == "reset":
            error = _reset_decisions(meeting)
        elif action == "save":
            error = _save_decision_edits(request, meeting)
        elif action == "publish":
            # 공개 버튼이 화면의 선택을 함께 저장한 뒤 공개한다. 저장과 공개가 나뉘어 있어서
            # 라디오를 다 고르고 공개를 눌렀는데 "처리하지 않은 항목이 있습니다"가 나오는 것이
            # 진행자가 가장 자주 걸리던 지점이었다. 저장이 실패하면 저장된 만큼은 남기고
            # 사유를 보여준다('저장' 버튼은 중간 저장용으로 그대로 둔다).
            error = _save_decision_edits(request, meeting, publishing=True) or _publish_decisions(meeting)
        elif action == "unpublish":
            error = _unpublish_decisions(meeting)
        elif action == "revise":
            error = _revise_final_text(request, meeting)
        if error is None:
            return redirect("decision_review", meeting_id=meeting.id)

    compiling = tasks.decision_compile_state(meeting.id)["running"]
    if error is None and not compiling:
        # 백그라운드 추출이 실패하고 끝났으면 사유를 한 번 보여준다.
        error = tasks.pop_decision_compile_error(meeting.id) or None
    decisions = list(meeting.decisions.prefetch_related("interpretations"))
    rows = []
    for decision in decisions:
        interpretations = list(decision.interpretations.all())
        choice, custom_text = _current_choice(decision, interpretations)
        for interpretation in interpretations:
            # 라디오 value와 선택 여부는 템플릿 필터로 만들기 까다로워서(접두사 + 정수 id)
            # 여기서 붙여 둔다. 저장되지 않는 임시 속성이다.
            interpretation.choice_value = "%s%d" % (
                decision_compiler.CHOICE_CANDIDATE_PREFIX, interpretation.id,
            )
            interpretation.chosen = choice == interpretation.choice_value
            # AI가 짐작으로 채운 자리만 색으로 칠한다. 원문·회의정보에서 가져온 값은 표시하지
            # 않는다 — 전부 칠하면 아무것도 칠하지 않은 것과 같다.
            guessed = decision_checks.guessed_fills(interpretation.fills)
            interpretation.segments = decision_checks.fill_segments(
                interpretation.text, [fill["value"] for fill in guessed],
            )
            # 문장에서 찾지 못한 값으로는 경고하지 않는다 — 진행자가 어디를 봐야 할지 알 수 없다.
            marked = {segment["text"] for segment in interpretation.segments if segment["estimated"]}
            interpretation.estimate_gaps = list(dict.fromkeys(
                fill["gap"] for fill in guessed if fill["value"] in marked and fill["gap"]
            ))
        rows.append({
            "decision": decision,
            "interpretations": interpretations,
            "choice": choice,
            "custom_text": custom_text,
            # 근거 한 문장만으론 판단하기 어려울 때 앞뒤 맥락을 더 보여준다.
            "evidence_context": decision_compiler.context_around(meeting.transcript, decision.evidence),
            # 경고에 찍힌 표현(예: '그쪽')이 원문 어디서 나왔는지. 근거 문장 위치와 다를 수
            # 있어 따로 찾는다 — warning_context 참고.
            "warning_rows": [
                {"text": warning, "context": decision_compiler.warning_context(meeting.transcript, warning)}
                for warning in decision.warnings
            ],
            # 확인이 필요한데 아직 고르지 않은 항목 — 이대로 공개하면 빠진다.
            "blocking": decision.blocks_publish,
        })
    # 확인이 필요한 항목과 그대로 공개될 항목을 따로 묶는다. AI 모호 판정은 warnings가 아니라
    # divergence_reason에 담기므로 기준은 has_warning이다.
    # '다음 회의로 넘김'을 고른 항목은 공개되지 않고 이월된다. 세 묶음으로 나누지 않으면
    # 화면은 "그대로 공개된다"고 말하고 서버는 "넘긴다"로 처리해 서로 다른 말을 한다.
    carried_rows = [row for row in rows if row["decision"].marked_undecided]
    ambiguous_rows = [
        row for row in rows
        if row["decision"].needs_host and not row["decision"].marked_undecided
    ]
    clear_rows = [
        row for row in rows
        if not row["decision"].needs_host and not row["decision"].marked_undecided
    ]
    for group, group_rows in (("ambiguous", ambiguous_rows), ("clear", clear_rows), ("carried", carried_rows)):
        for row in group_rows:
            row["group"] = group
    # 한 문항씩 넘겨 보는 것은 진행자가 손대야 하는 항목뿐이다 — 확인이 필요한 것, 그다음 넘길 것.
    # 그대로 공개될 항목은 고를 것이 없으므로 화면에서 접힌 묶음으로 내린다(펼치면 다 보인다).
    # 번호는 접힌 묶음까지 이어 붙여 "결정 N개"와 맞춘다.
    steps = ambiguous_rows + carried_rows
    for number, row in enumerate(steps + clear_rows, start=1):
        row["num"] = number
    drafts = [row for row in rows if row["decision"].status == Decision.STATUS_DRAFT]
    blockers = [row for row in drafts if row["blocking"]]
    return render(request, "meetalign/decision_review.html", {
        "meeting": meeting,
        "rows": rows,
        "steps": steps,
        "error": error,
        "compiled": bool(decisions),
        # 추출 중이면 화면이 스스로 새로 고친다. 결정이 저장되는 즉시 끝난 것으로 본다.
        "compiling": compiling and not decisions,
        "ambiguous_rows": ambiguous_rows,
        "clear_rows": clear_rows,
        "carried_rows": carried_rows,
        "ambiguous_count": len(ambiguous_rows),
        # 확인창이 "공개 N개 · 이월 M개"를 정확히 말할 수 있게 draft만 나눠 센다.
        "publish_count": sum(
            1 for row in drafts if not row["decision"].marked_undecided),
        "carried_count": sum(
            1 for row in drafts if row["decision"].marked_undecided),
        "draft_count": len(drafts),
        "blocker_count": len(blockers),
        # 고르지 않은 항목이 있어도 공개할 수 있다 — 그 항목은 공개할 때 빠진다.
        "can_publish": bool(drafts),
        "published_count": sum(1 for row in rows if row["decision"].status == Decision.STATUS_PUBLISHED),
        "undecided_count": sum(1 for row in rows if row["decision"].status == Decision.STATUS_UNDECIDED),
        # 공개 취소는 전부 공개된 뒤에만 — 잘못 누른 공개를 수습하는 통로다.
        "can_unpublish": bool(decisions) and not drafts,
        "evidence_data": {
            str(row["decision"].id): _review_evidence_spans(meeting, row["decision"], row["interpretations"])
            for row in rows
        },
        # 보고 있던 문항 위치를 기억하는 브라우저 저장 키 — 저장·공개 뒤 같은 문항으로 돌아온다.
        "step_key": "decision-step-review-%d" % meeting.id,
        "choice_candidate_prefix": decision_compiler.CHOICE_CANDIDATE_PREFIX,
        "choice_custom": decision_compiler.CHOICE_CUSTOM,
        "choice_undecided": decision_compiler.CHOICE_UNDECIDED,
        "choice_drop": decision_compiler.CHOICE_DROP,
        "editable": all(row["decision"].status == Decision.STATUS_DRAFT for row in rows),
        "status_draft": Decision.STATUS_DRAFT,
        "status_published": Decision.STATUS_PUBLISHED,
        "status_undecided": Decision.STATUS_UNDECIDED,
    })


def _evidence_context(decision, interpretations):
    """같은 근거 문장이 원문에 여러 번 나올 때 어느 자리인지 가리는 데 쓰는 그 결정의 내용."""
    return " ".join([decision.what, decision.final_text] + [interpretation.text for interpretation in interpretations])


def _review_evidence_spans(meeting, decision, interpretations):
    """진행자 화면의 녹취 원문 창에서 하이라이트할 구간 — 후보마다의 근거와 명확한 결정의 근거."""
    evidences = [("후보 %s" % interpretation.label, interpretation.evidence) for interpretation in interpretations]
    evidences.append(("근거", decision.evidence))
    return decision_compiler.evidence_spans(meeting.transcript, evidences, _evidence_context(decision, interpretations))


def _published_evidence_spans(meeting, decision):
    """팀원 화면의 하이라이트 구간 — 공개된 문장의 근거.

    진행자가 후보를 골랐으면 그 후보의 근거, 명확한 결정이면 저장된 근거를 쓴다. 진행자가 직접
    쓴 문장처럼 딱 맞는 근거가 없으면 후보들의 근거를 모두 보여준다(그 결정이 나온 대목이다).
    """
    interpretations = list(decision.interpretations.all())
    chosen = [i for i in interpretations if i.text == decision.final_text and i.evidence]
    if chosen:
        evidences = [("", chosen[0].evidence)]
    elif decision.evidence:
        evidences = [("", decision.evidence)]
    else:
        evidences = [("", interpretation.evidence) for interpretation in interpretations]
    return decision_compiler.evidence_spans(meeting.transcript, evidences, _evidence_context(decision, interpretations))


def _published_evidence(decision):
    """카드 안에 접어 둘 근거 발언 한 문장 — 진행자가 고른 후보의 근거, 없으면 저장된 근거.

    _published_evidence_spans(녹취 원문 창의 하이라이트 위치)와 같은 기준으로 고른다.
    """
    for interpretation in decision.interpretations.all():
        if interpretation.text == decision.final_text and interpretation.evidence:
            return interpretation.evidence
    return decision.evidence


def _ack_rows(user, meeting):
    """팀원 결정 확인 화면의 카드 묶음과 '내' 진행률.

    담는 것은 내 선택·내 의견과, 결정마다 올라온 익명 질문이다 — 누가 무엇을 골랐는지는 저장만 하고
    어떤 화면에도 내보내지 않으므로 다른 사람의 선택·의견은 여기서 아예 조회하지 않는다. 익명 질문은
    익명 대화창처럼 팀 전체 공개이고 작성자는 드러내지 않는다.

    진행률의 분모는 '확인할 수 있는 항목'(공개된 항목)이다. 미결정으로 이월된 항목은 읽기
    전용이라 분모에서 빠진다 — 아무리 눌러도 줄지 않는 숫자가 되기 때문이다.
    """
    decisions = decision_compiler.published_decisions(meeting)
    # 읽기 전용인 미결정 항목은 뒤로 몰아 둔다. 쪽을 넘기며 확인하는 화면이라, 손댈 것이 없는
    # 카드가 중간에 끼면 흐름이 끊긴다. 같은 묶음 안에서는 원래 순서를 그대로 둔다(정렬이 안정적).
    decisions.sort(key=lambda decision: decision.status != Decision.STATUS_PUBLISHED)
    mine = {
        agreement.decision_id: agreement
        for agreement in DecisionAgreement.objects.filter(decision__in=decisions, user=user)
    }
    # 이 결정에 대해 올라온 익명 질문과 주최자 답변 — 익명 대화창과 같은 기준으로 팀원 모두에게 보인다.
    # 작성자는 내보내지 않고, 내 질문인지만 화면에서 표시한다(question.user_id).
    asked = {}
    for question in Question.objects.filter(
        meeting=meeting, decision__in=decisions,
    ).order_by("created_at", "id"):
        asked.setdefault(question.decision_id, []).append(question)
    rows = []
    for number, decision in enumerate(decisions, start=1):
        agreement = mine.get(decision.id)
        rows.append({
            "number": number,
            "decision": decision,
            "agreement": agreement,
            "kind": agreement.kind if agreement else "",
            # 공개 뒤 확정 문장이 고쳐지면 내 확인 기록도 지워진다(decision_compiler.revise_final_text).
            # 그 자리를 빈칸으로 두면 왜 다시 눌러야 하는지 알 수 없으므로 사유를 표시한다.
            "needs_recheck": decision.revised and agreement is None,
            "answerable": decision.status == Decision.STATUS_PUBLISHED,
            "evidence": _published_evidence(decision),
            "questions": asked.get(decision.id, []),
        })
    total = sum(1 for row in rows if row["answerable"])
    done = sum(1 for row in rows if row["answerable"] and row["agreement"])
    progress = {"done": done, "total": total, "percent": int(done * 100 / total) if total else 0}
    return rows, progress


# 한 칸에 보여줄 카드 수. 일곱 개가 한꺼번에 쏟아지면 어디까지 봤는지 놓치기 때문에 끊어 보여준다.
ACK_PAGE_SIZE = 4


def _ack_page_number(request):
    """보고 있던 칸 번호. 저장·질문 뒤에도 같은 칸으로 돌아오게 폼이 함께 보낸다."""
    try:
        return max(1, int(request.POST.get("page") or request.GET.get("page") or 1))
    except (TypeError, ValueError):
        return 1


def _ack_context(request, meeting):
    rows, progress = _ack_rows(request.user, meeting)
    # 진행률은 칸과 무관하게 이 회의 전체 기준이다 — 칸마다 분모가 바뀌면 얼마나 남았는지 알 수 없다.
    page = Paginator(rows, ACK_PAGE_SIZE).get_page(_ack_page_number(request))
    shown = list(page)
    return {
        "meeting": meeting,
        "rows": shown,
        "page": page,
        "ack_page": page.number,
        "progress": progress,
        "evidence_data": {
            str(row["decision"].id): _published_evidence_spans(meeting, row["decision"]) for row in shown
        },
        # 카드 목록이라 한 문항씩 넘겨 보지 않는다 — 녹취 원문 창만 쓴다(_decision_viewer).
        "unpaged": True,
        "status_published": Decision.STATUS_PUBLISHED,
        "kind_ack": DecisionAgreement.KIND_ACK,
        "kind_differ": DecisionAgreement.KIND_DIFFER,
    }


@login_required
def decision_ack(request, meeting_id):
    """팀원 화면. 공개된 결정 전체를 한 화면 카드 목록으로 보고, 카드마다 확인 여부를 남긴다.

    저장은 카드별 전용 엔드포인트(decision_agree)가 맡는다 — 이 화면은 GET만 받는다.
    이 화면을 열면 이 회의의 결정 알림은 읽은 것으로 보고 지운다. 진행자는 진행자 화면으로
    보낸다. 공개 전(draft) 항목은 published_decisions()가 아예 돌려주지 않으므로 응답에 담기지 않는다.
    """
    meeting = _decision_meeting(request, meeting_id)
    if meeting.host_id == request.user.id:
        return redirect("decision_review", meeting_id=meeting.id)
    notifications.mark_read(request.user, meeting)
    return render(request, "meetalign/decision_ack.html", _ack_context(request, meeting))


def _ack_redirect(request, meeting, decision):
    """자바스크립트가 꺼진 브라우저용 — 보고 있던 칸의 그 카드 자리로 되돌려 보낸다."""
    return redirect("%s?page=%d#decision-%d" % (
        reverse("decision_ack", args=[meeting.id]), _ack_page_number(request), decision.id,
    ))


def _wants_card_json(request):
    """카드 하나만 갈아끼우려는 fetch 요청인가 — 스크립트가 붙이는 표시로 일반 form POST와 가른다."""
    return request.headers.get("x-requested-with") == "fetch"


def _ack_response(request, meeting, decision, **card_extra):
    """카드 하나를 처리한 뒤의 응답. 그려지는 카드는 어느 경로에서나 _decision_card.html 하나다.

    - fetch: 그 카드 조각과 진행률만 JSON으로 — 전체 새로고침 없이 그 카드만 갈아끼운다.
    - 스크립트가 꺼진 브라우저: 보여줄 것(AI 답변·오류)이 있으면 화면을 바로 그리고,
      없으면 방금 누른 카드 자리로 되돌려 보낸다. 리다이렉트로는 방금 받은 답변이 사라진다.
    """
    if _wants_card_json(request):
        rows, progress = _ack_rows(request.user, meeting)
        row = next((row for row in rows if row["decision"].id == decision.id), None)
        if row is None:  # 응답을 만드는 사이에 진행자가 공개를 취소한 경우.
            raise Http404
        row.update(card_extra)
        card = render_to_string("meetalign/_decision_card.html", {
            "meeting": meeting, "row": row, "ack_page": _ack_page_number(request),
            "kind_ack": DecisionAgreement.KIND_ACK, "kind_differ": DecisionAgreement.KIND_DIFFER,
        }, request=request)
        return JsonResponse({"card": card, "progress": progress})
    if not card_extra:
        return _ack_redirect(request, meeting, decision)
    context = _ack_context(request, meeting)
    for row in context["rows"]:
        if row["decision"].id == decision.id:
            row.update(card_extra)
    return render(request, "meetalign/decision_ack.html", context)


@login_required
def decision_agree(request, meeting_id, decision_id):
    """카드 하나의 '확인했어요 / 달랐어요'를 저장한다.

    공개된 항목만 대상이다 — 미결정으로 이월된 항목은 404로 막는다(화면에도 버튼이 없다).
    선택은 다시 바꿀 수 있어야 하므로 update_or_create로 (항목, 사용자)당 한 줄만 남긴다.

    DecisionAgreement.comment(한 줄 의견)는 화면에서 뺐다 — 두 버튼만 남겼고, 그래서 이 뷰는
    항상 빈 문자열로 맞춘다. 칸은 모델에 그대로 두되(옛 기록 보존) 화면과 어긋나지 않게 한다.
    """
    meeting = _decision_meeting(request, meeting_id)
    decision = get_object_or_404(
        Decision, pk=decision_id, meeting=meeting, status=Decision.STATUS_PUBLISHED,
    )
    if request.method != "POST":
        return _ack_redirect(request, meeting, decision)
    kind = request.POST.get("kind", "")
    if kind not in (DecisionAgreement.KIND_ACK, DecisionAgreement.KIND_DIFFER):
        return HttpResponseBadRequest("알 수 없는 선택입니다.")
    DecisionAgreement.objects.update_or_create(
        decision=decision, user=request.user,
        defaults={"kind": kind, "comment": ""},
    )
    return _ack_response(request, meeting, decision)


# 카드에서 AI에게 물을 때 질문 앞에 붙이는 맥락 한 줄. 대화창 답변 함수는 회의록만 근거로
# 답하므로, 어떤 결정에 대한 질문인지는 이렇게 질문 안에 담아 전달한다.
ASK_DECISION_CONTEXT = "이 질문은 다음 결정에 대한 것이다: %s"


@login_required
def decision_ask(request, meeting_id, decision_id):
    """결정 카드 안에서 바로 묻기. 대화창으로 떠나지 않아도 질문과 답변이 끝나게 한다.

    mode=anon이면 주최자에게 보내는 익명 질문(Question)으로 저장하고 어떤 결정에 대한
    질문인지만 함께 남긴다 — 작성자는 어느 화면에도 나가지 않는다.

    그 밖에는 AI에게 묻는다. 답변은 대화창과 같은 함수(services.chat_reply)로 만든다 —
    회의록만 근거로 답하고 회의에 없는 내용은 답하지 않는 규칙이 그대로 적용된다. 질문과
    답변은 대화창 기록(ChatMessage)에도 남겨 대화창에서 이어 볼 수 있게 한다.
    """
    meeting = _decision_meeting(request, meeting_id)
    decision = get_object_or_404(
        Decision, pk=decision_id, meeting=meeting,
        status__in=[Decision.STATUS_PUBLISHED, Decision.STATUS_UNDECIDED],
    )
    if request.method != "POST":
        return _ack_redirect(request, meeting, decision)
    text = request.POST.get("text", "").strip()
    if not text:
        return _ack_response(request, meeting, decision, ask_open=True, ask_error="질문을 입력해 주세요.")
    if request.POST.get("mode") == "anon":
        if meeting.ended:
            return _ack_response(
                request, meeting, decision,
                ask_open=True, ask_error="종료된 회의에는 익명 질문을 보낼 수 없습니다.",
            )
        # 대화창의 익명 질문과 같은 Question에 저장하고, 어떤 결정에 대한 질문인지만 함께 남긴다.
        # 작성자는 주최자 화면에도 다른 팀원 화면에도 나가지 않는다(inbox / answers).
        Question.objects.create(meeting=meeting, user=request.user, text=text, decision=decision)
        return _ack_response(request, meeting, decision, ask_open=True, ask_sent=True)
    asked = "%s\n%s" % (ASK_DECISION_CONTEXT % decision.published_text, text)
    ChatMessage.objects.create(meeting=meeting, user=request.user, role=ChatMessage.ROLE_USER, text=asked)
    reply = services.chat_reply(meeting, asked)
    ChatMessage.objects.create(meeting=meeting, user=request.user, role=ChatMessage.ROLE_ASSISTANT, text=reply)
    return _ack_response(request, meeting, decision, ask_open=True, ai_question=text, ai_answer=reply)


@login_required
def notification_list(request):
    """상단 메뉴 알림 아이콘이 여는 알림 목록. 읽은 알림은 지워지므로 여기 보이는 것은 모두 새 알림이다.

    '모두 읽음'은 그 사람의 알림을 한 번에 지운다.
    """
    if request.method == "POST" and request.POST.get("action") == "read_all":
        notifications.clear_all(request.user)
        return redirect("notification_list")
    items = list(
        Notification.objects.filter(user=request.user, read_at__isnull=True, meeting__team__members=request.user)
        .select_related("meeting", "meeting__team")[:50]
    )
    return render(request, "meetalign/notifications.html", {"items": items})
