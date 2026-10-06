import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Answer, CheckQuestion, Meeting, Question, Team


class AuthTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        r = self.client.post(reverse("signup"), {
            "username": "newuser", "password1": "a-str0ng-pw", "password2": "a-str0ng-pw",
        })
        self.assertRedirects(r, reverse("index"))
        self.assertTrue(User.objects.filter(username="newuser").exists())
        self.assertIn("_auth_user_id", self.client.session)

    def test_signup_rejects_mismatched_passwords(self):
        r = self.client.post(reverse("signup"), {
            "username": "newuser", "password1": "a-str0ng-pw", "password2": "different",
        })
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username="newuser").exists())

    def test_login_with_correct_credentials_succeeds(self):
        User.objects.create_user("a", password="pw12345!")
        r = self.client.post(reverse("login"), {"username": "a", "password": "pw12345!"})
        self.assertRedirects(r, reverse("index"))
        self.assertIn("_auth_user_id", self.client.session)

    def test_login_with_wrong_password_fails(self):
        User.objects.create_user("a", password="pw12345!")
        r = self.client.post(reverse("login"), {"username": "a", "password": "wrong"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_logout_clears_session(self):
        user = User.objects.create_user("a", password="pw12345!")
        self.client.force_login(user)
        self.client.get(reverse("logout"))
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_protected_pages_redirect_to_login(self):
        team = Team.objects.create(name="t")
        r = self.client.get(reverse("meeting_list", args=[team.id]))
        self.assertRedirects(r, "%s?next=%s" % (reverse("login"), reverse("meeting_list", args=[team.id])))


class PrototypeFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("a", password="pw12345!")
        self.client.force_login(self.user)
        self.team = Team.objects.create(name="t")
        self.team.members.add(self.user)
        self.meeting = Meeting.objects.create(team=self.team, title="m")

    def test_index_public(self):
        self.assertEqual(self.client.get(reverse("index")).status_code, 200)

    def test_team_create_and_join(self):
        self.client.post(reverse("team_join"), {"action": "create", "name": "new"})
        self.assertTrue(Team.objects.get(name="new").members.filter(pk=self.user.pk).exists())
        other_team = Team.objects.create(name="o")
        self.client.post(reverse("team_join"), {"action": "join", "name": "o"})
        self.assertTrue(other_team.members.filter(pk=self.user.pk).exists())

    def test_non_member_forbidden(self):
        meeting = Meeting.objects.create(team=Team.objects.create(name="other"), title="x")
        self.assertEqual(self.client.get(reverse("meeting_detail", args=[meeting.id])).status_code, 404)

    def test_meeting_create_and_all_screens(self):
        self.client.post(reverse("meeting_create", args=[self.team.id]), {"title": "x"})
        self.assertTrue(Meeting.objects.filter(title="x").exists())
        for name in ["meeting_list", "meeting_create"]:
            self.assertEqual(self.client.get(reverse(name, args=[self.team.id])).status_code, 200)
        for name in ["meeting_detail", "meeting_result", "chat", "inbox",
                     "summary", "answers", "understanding"]:
            self.assertEqual(self.client.get(reverse(name, args=[self.meeting.id])).status_code, 200, name)

    def test_submit_and_end(self):
        url = reverse("meeting_detail", args=[self.meeting.id])
        self.assertRedirects(self.client.post(url, {"action": "submit"}),
                             reverse("meeting_result", args=[self.meeting.id]))
        self.assertRedirects(self.client.post(url, {"action": "end"}),
                             reverse("answers", args=[self.meeting.id]))
        self.meeting.refresh_from_db()
        self.assertTrue(self.meeting.ended)

    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    def test_recording_upload_is_saved(self):
        url = reverse("meeting_detail", args=[self.meeting.id])
        file = SimpleUploadedFile("test.mp3", b"fake audio bytes", content_type="audio/mpeg")
        self.client.post(url, {"action": "submit", "recording": file})
        self.meeting.refresh_from_db()
        self.assertTrue(self.meeting.recording)
        self.assertIn("test", self.meeting.recording.name)

    def test_anonymous_question_and_answer(self):
        self.client.post(reverse("chat", args=[self.meeting.id]), {"mode": "anon", "text": "why?"})
        q = Question.objects.get()
        self.client.post(reverse("answers", args=[self.meeting.id]), {"question_id": q.id, "answer": "because"})
        q.refresh_from_db()
        self.assertEqual(q.answer, "because")

    def test_llm_chat_does_not_store(self):
        r = self.client.post(reverse("chat", args=[self.meeting.id]), {"mode": "llm", "text": "hi"})
        self.assertContains(r, "가짜 LLM")
        self.assertEqual(Question.objects.count(), 0)

    def test_meeting_detail_back_link(self):
        r = self.client.get(reverse("meeting_detail", args=[self.meeting.id]))
        self.assertContains(r, 'href="%s"' % reverse("meeting_list", args=[self.team.id]))

    def test_sub_screens_have_back_to_meeting_link(self):
        back = 'href="%s"' % reverse("meeting_detail", args=[self.meeting.id])
        for name in ["meeting_result", "chat", "inbox", "summary", "answers", "understanding"]:
            r = self.client.get(reverse(name, args=[self.meeting.id]))
            self.assertContains(r, back, msg_prefix=name)

    def test_understanding_creates_fixed_questions_once(self):
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        self.assertEqual(CheckQuestion.objects.filter(meeting=self.meeting).count(), 2)

    def test_understanding_saves_and_prefills_answer(self):
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        q = CheckQuestion.objects.filter(meeting=self.meeting).first()
        self.client.post(reverse("understanding", args=[self.meeting.id]), {"q%d" % q.id: "내 답변"})
        self.assertEqual(Answer.objects.get(question=q, user=self.user).text, "내 답변")
        r = self.client.get(reverse("understanding", args=[self.meeting.id]))
        self.assertContains(r, "내 답변")

    def test_understanding_resubmit_updates_not_duplicates(self):
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        q = CheckQuestion.objects.filter(meeting=self.meeting).first()
        self.client.post(reverse("understanding", args=[self.meeting.id]), {"q%d" % q.id: "첫 답변"})
        self.client.post(reverse("understanding", args=[self.meeting.id]), {"q%d" % q.id: "수정된 답변"})
        self.assertEqual(Answer.objects.filter(question=q, user=self.user).count(), 1)
        self.assertEqual(Answer.objects.get(question=q, user=self.user).text, "수정된 답변")

    def test_same_question_can_hold_different_users_answers(self):
        # 핵심 검증: 같은 질문에 팀원마다 답이 따로 저장돼야 이해도 비교가 가능하다.
        other = User.objects.create_user("b")
        question = CheckQuestion.objects.create(meeting=self.meeting, text="Q")
        Answer.objects.create(question=question, user=self.user, text="A의 이해")
        Answer.objects.create(question=question, user=other, text="B의 이해")
        self.assertEqual(question.answers.count(), 2)

    def test_discrepancy_flagged_when_answers_differ(self):
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        other = User.objects.create_user("b")
        question = CheckQuestion.objects.filter(meeting=self.meeting).first()
        Answer.objects.create(question=question, user=self.user, text="A안")
        Answer.objects.create(question=question, user=other, text="B안")
        r = self.client.get(reverse("understanding", args=[self.meeting.id]))
        self.assertContains(r, "b: B안")
        self.assertContains(r, "a: A안")
        self.assertNotContains(r, "엇갈린 답변이 없습니다.")

    def test_no_discrepancy_when_answers_match(self):
        self.client.get(reverse("understanding", args=[self.meeting.id]))
        other = User.objects.create_user("b")
        question = CheckQuestion.objects.filter(meeting=self.meeting).first()
        Answer.objects.create(question=question, user=self.user, text="같은 답")
        Answer.objects.create(question=question, user=other, text="같은 답")
        r = self.client.get(reverse("understanding", args=[self.meeting.id]))
        self.assertContains(r, "엇갈린 답변이 없습니다.")
