import datetime
from django.test import TestCase
from django.utils import timezone
from django.urls import reverse
from .models import Question


def create_question(question_text, days):
    """create question. days is negative for publishing date in the past"""
    time = ((timezone.now()) + (datetime.timedelta(days=days)))
    return Question.objects.create(question_text=question_text, pub_date=time)


class QuestionDetailViewTests(TestCase):
    def test_past_question(self):
        q = create_question(question_text="past question", days=-5)
        url = reverse("polls:detail", args=(q.id, ))
        response = self.client.get(url)
        self.assertContains(response, q.question_text)

    def test_future_question(self):
        q = create_question(question_text="future question", days=5)
        url = reverse("polls:detail", args=(q.id, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)


class QuestionIndexViewTests(TestCase):
    def test_no_question(self):
        # no pre
        response = self.client.get(reverse("polls:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "no polls are available")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_past_question(self):
        q = create_question(question_text="past question", days=-30)
        response = self.client.get(reverse("polls:index"))
        self.assertQuerysetEqual(response.context["latest_question_list"], [q])

    def test_future_question(self):
        q = create_question(question_text="futurue question", days=30)
        response = self.client.get(reverse("polls:index"))
        self.assertContains(response, "no polls are available")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_future_and_past_question(self):
        q0 = create_question(question_text="past question", days=-30)
        q1 = create_question(question_text="futurue question", days=30)
        response = self.client.get(reverse("polls:index"))
        self.assertQuerysetEqual(response.context["latest_question_list"],
                                 [q0])

    def test_two_past_question(self):
        q0 = create_question(question_text="past question", days=-30)
        q1 = create_question(question_text="futurue question", days=-5)
        response = self.client.get(reverse("polls:index"))
        self.assertQuerysetEqual(response.context["latest_question_list"],
                                 [q1, q0])


class QuestionModelTests(TestCase):
    def test_was_published_recently_with_future_question(self):
        time = ((timezone.now()) + (datetime.timedelta(days=30)))
        question = Question(pub_date=time)
        self.assertIs(question.was_published_recently(), False)

    def test_was_published_recently_with_old_question(self):
        time = ((timezone.now()) - (datetime.timedelta(days=1, seconds=1)))
        question = Question(pub_date=time)
        self.assertIs(question.was_published_recently(), False)

    def test_was_published_recently_with_recent_question(self):
        time = ((timezone.now()) -
                (datetime.timedelta(hours=23, minutes=59, seconds=59)))
        question = Question(pub_date=time)
        self.assertIs(question.was_published_recently(), True)
