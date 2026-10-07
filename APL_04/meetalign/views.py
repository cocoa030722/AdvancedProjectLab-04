from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render

from .models import Meeting, Question, Team


def current_user():
    # 프로토타입: 이미 로그인한 사용자가 있다고 가정하고 가장 먼저 만든 사용자를 사용한다.
    return User.objects.order_by("pk").first() or User.objects.create_user("demo")


def index(request):
    user = current_user()
    return render(request, "meetalign/index.html", {"user": user, "teams": user.teams.all()})


# 로그인 관련 페이지는 화면만 남기고 실제 인증은 하지 않는다.
def signup(request):
    if request.method == "POST":
        return redirect("index")
    return render(request, "meetalign/signup.html")


def login_view(request):
    if request.method == "POST":
        return redirect("index")
    return render(request, "meetalign/login.html")


def logout_view(request):
    return redirect("index")


def team_join(request):
    error = ""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        if request.POST.get("action") == "create":
            if not name or Team.objects.filter(name=name).exists():
                error = "팀 이름이 비었거나 이미 존재합니다."
            else:
                Team.objects.create(name=name).members.add(current_user())
                return redirect("index")
        else:
            team = Team.objects.filter(name=name).first()
            if team is None:
                error = "해당 이름의 팀이 없습니다."
            else:
                team.members.add(current_user())
                return redirect("meeting_list", team_id=team.id)
    return render(request, "meetalign/team_join.html", {"error": error})


def _team(request, team_id):
    return get_object_or_404(Team, pk=team_id, members=current_user())


def _meeting(request, meeting_id):
    return get_object_or_404(Meeting, pk=meeting_id, team__members=current_user())


def meeting_list(request, team_id):
    team = _team(request, team_id)
    return render(request, "meetalign/meeting_list.html", {"team": team, "meetings": team.meetings.all()})


def meeting_create(request, team_id):
    team = _team(request, team_id)
    if request.method == "POST" and request.POST.get("title", "").strip():
        meeting = Meeting.objects.create(team=team, title=request.POST["title"].strip())
        return redirect("meeting_detail", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_create.html", {"team": team})


def meeting_detail(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        # 녹음 파일은 저장/처리하지 않는다 (목업)
        if request.POST.get("action") == "end":
            meeting.ended = True
            meeting.save()
            return redirect("answers", meeting_id=meeting.id)
        return redirect("meeting_result", meeting_id=meeting.id)
    return render(request, "meetalign/meeting_detail.html", {"meeting": meeting})


def meeting_result(request, meeting_id):
    return render(request, "meetalign/meeting_result.html", {"meeting": _meeting(request, meeting_id)})


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


def inbox(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    return render(request, "meetalign/inbox.html", {"meeting": meeting, "questions": meeting.questions.all()})


def summary(request, meeting_id):
    return render(request, "meetalign/summary.html", {"meeting": _meeting(request, meeting_id)})


def answers(request, meeting_id):
    meeting = _meeting(request, meeting_id)
    if request.method == "POST":
        question = get_object_or_404(Question, pk=request.POST.get("question_id"), meeting=meeting)
        question.answer = request.POST.get("answer", "").strip()
        question.save()
        return redirect("answers", meeting_id=meeting.id)
    return render(request, "meetalign/answers.html", {"meeting": meeting, "questions": meeting.questions.all()})


def understanding(request, meeting_id):
    return render(request, "meetalign/understanding.html", {"meeting": _meeting(request, meeting_id)})
