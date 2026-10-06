from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.shortcuts import get_object_or_404, redirect, render

from .models import Answer, CheckQuestion, Meeting, Question, Team

# LLM이 실제로 문제를 내기 전까지 쓰는 고정 문항 (구현사항.txt: 실제 기능 구현 안 함)
FAKE_CHECK_QUESTIONS = [
    "이번 회의의 핵심 결정 사항은?",
    "내가 맡은 다음 할 일은?",
]


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
            return redirect(request.GET.get("next") or "index")
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
        name = request.POST.get("name", "").strip()
        if request.POST.get("action") == "create":
            if not name or Team.objects.filter(name=name).exists():
                error = "팀 이름이 비었거나 이미 존재합니다."
            else:
                Team.objects.create(name=name).members.add(request.user)
                return redirect("index")
        else:
            team = Team.objects.filter(name=name).first()
            if team is None:
                error = "해당 이름의 팀이 없습니다."
            else:
                team.members.add(request.user)
                return redirect("meeting_list", team_id=team.id)
    return render(request, "meetalign/team_join.html", {"error": error})


def _team(request, team_id):
    return get_object_or_404(Team, pk=team_id, members=request.user)


def _meeting(request, meeting_id):
    return get_object_or_404(Meeting, pk=meeting_id, team__members=request.user)


@login_required
def meeting_list(request, team_id):
    team = _team(request, team_id)
    return render(request, "meetalign/meeting_list.html", {"team": team, "meetings": team.meetings.all()})


@login_required
def meeting_create(request, team_id):
    team = _team(request, team_id)
    if request.method == "POST" and request.POST.get("title", "").strip():
        deadline = request.POST.get("deadline") or None
        meeting = Meeting.objects.create(team=team, title=request.POST["title"].strip(), deadline=deadline)
        return redirect("meeting_detail", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_create.html", {"team": team})


@login_required
def meeting_detail(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        recording = request.FILES.get("recording")
        if recording:
            meeting.recording = recording
            meeting.save()
        # STT/LLM 처리는 아직 없음 (다음 작업 대상). 파일만 실제로 저장한다.
        if request.POST.get("action") == "end":
            meeting.ended = True
            meeting.save()
            return redirect("answers", meeting_id=meeting.id)
        return redirect("meeting_result", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_detail.html", {"meeting": meeting})


@login_required
def meeting_result(request, meeting_id):
    return render(request, "meetalign/meeting_result.html", {"meeting": _meeting(request, meeting_id)})


@login_required
def chat(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    mode = request.POST.get("mode") or request.GET.get("mode", "llm")
    reply = ""
    if request.method == "POST" and request.POST.get("text", "").strip():
        text = request.POST["text"].strip()
        if mode == "anon":
            Question.objects.create(meeting=meeting, text=text)
        else:
            reply = "(가짜 LLM 응답) '%s'에 대한 답변입니다." % text
    return render(request, "meetalign/chat.html", {"meeting": meeting, "mode": mode, "reply": reply})


@login_required
def inbox(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    return render(request, "meetalign/inbox.html", {"meeting": meeting, "questions": meeting.questions.all()})


@login_required
def summary(request, meeting_id):
    return render(request, "meetalign/summary.html", {"meeting": _meeting(request, meeting_id)})


@login_required
def answers(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        question = get_object_or_404(Question, pk=request.POST.get("question_id"), meeting=meeting)
        question.answer = request.POST.get("answer", "").strip()
        question.save()
        return redirect("answers", meeting_id=meeting.id)
    return render(request, "meetalign/answers.html", {"meeting": meeting, "questions": meeting.questions.all()})


def _find_discrepancies(meeting):
    # 지금은 답변 문자열이 다르면 불일치로 본다. LLM 의미 비교로 교체할 자리.
    found = []
    for question in meeting.check_questions.all():
        answers = list(question.answers.select_related("user"))
        if len({a.text.strip() for a in answers}) > 1:
            found.append((question, answers))
    return found


@login_required
def understanding(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if not meeting.check_questions.exists():
        for order, text in enumerate(FAKE_CHECK_QUESTIONS):
            CheckQuestion.objects.create(meeting=meeting, text=text, order=order)
    user = request.user
    questions = list(meeting.check_questions.all())
    if request.method == "POST":
        for question in questions:
            text = request.POST.get("q%d" % question.id, "").strip()
            if text:
                Answer.objects.update_or_create(question=question, user=user, defaults={"text": text})
        return redirect("understanding", meeting_id=meeting.id)
    my_answers = {a.question_id: a.text for a in Answer.objects.filter(question__meeting=meeting, user=user)}
    rows = [(q, my_answers.get(q.id, "")) for q in questions]
    return render(request, "meetalign/understanding.html", {
        "meeting": meeting,
        "rows": rows,
        "discrepancies": _find_discrepancies(meeting),
    })
