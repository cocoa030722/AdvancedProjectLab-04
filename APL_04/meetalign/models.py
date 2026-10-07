from django.conf import settings
from django.db import models


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

    def __str__(self):
        return self.title


class Question(models.Model):
    """회의별 대화방의 익명 질문. 작성자는 저장하지 않는다."""
    meeting = models.ForeignKey(Meeting, on_delete=models.CASCADE, related_name="questions")
    text = models.TextField()
    answer = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:30]
