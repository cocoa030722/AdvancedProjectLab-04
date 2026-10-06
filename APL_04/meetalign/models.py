from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)
    members = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="teams")

    def __str__(self):
        return self.name


class Meeting(models.Model):
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="meetings")
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    ended = models.BooleanField(default=False)
    deadline = models.DateField(null=True, blank=True)
    # STT/LLM 처리는 아직 없음. 지금은 파일을 실제로 저장하는 것까지만 한다.
    recording = models.FileField(upload_to="recordings/", blank=True, null=True)
    # 회의록(전사+요약). 지금은 저장 자리만 있고, 채우는 건 STT/LLM 작업에서 한다.
    record = models.TextField(blank=True)

    def __str__(self):
        return self.title

    @property
    def is_due_tomorrow(self):
        return self.deadline is not None and self.deadline - timezone.localdate() == timedelta(days=1)


class Question(models.Model):
    """회의별 대화방의 익명 질문. 작성자는 저장하지 않는다."""
    meeting = models.ForeignKey(Meeting, on_delete=models.CASCADE, related_name="questions")
    text = models.TextField()
    answer = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:30]


class CheckQuestion(models.Model):
    """이해도 검증을 위해 (LLM이) 낸 질문. 회의당 여러 개이며 팀원 전원이 각자 답한다."""
    meeting = models.ForeignKey(Meeting, on_delete=models.CASCADE, related_name="check_questions")
    text = models.TextField()
    order = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "id"]

    def __str__(self):
        return self.text[:30]


class Answer(models.Model):
    """CheckQuestion에 대한 팀원별 답변. 같은 질문이라도 사람마다 따로 저장되어야
    "누구와 누구가 다르게 이해했는지" 비교가 가능하다."""
    question = models.ForeignKey(CheckQuestion, on_delete=models.CASCADE, related_name="answers")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="check_answers")
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("question", "user")

    def __str__(self):
        return "%s - %s" % (self.user, self.question_id)
