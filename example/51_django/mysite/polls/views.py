from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from .models import Question, Choice


class IndexView(generic.ListView):
    template_name = "polls/index.html"
    context_object_name = "latest_question_list"

    def get_queryset(self):
        return Question.objects.filter(
            pub_date__lte=timezone.now()).order_by("-pub_date")[:5]


class DetailView(generic.DetailView):
    template_name = "polls/detail.html"
    model = Question

    def get_queryset(self):
        return Question.objects.filter(pub_date__lte=timezone.now())


class ResultsView(generic.DetailView):
    template_name = "polls/results.html"
    model = Question


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST["choice"])
    except (
            KeyError,
            Choice.DoesNotExist,
    ):
        # redisplay voting form
        return render(
            request, "polls/detail.html",
            dict(question=question,
                 error_message="you didn't select a choice"))
    else:
        selected_choice.votes = ((1) + (selected_choice.votes))
        selected_choice.save()
        return HttpResponseRedirect(
            reverse("polls:results", args=(question.id, )))
