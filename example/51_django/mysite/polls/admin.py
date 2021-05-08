from django.contrib import admin
from .models import Question, Choice


class ChoiceInline(admin.StackedInline):
    model = Choice
    extra = 3


class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [(
        None,
        dict(fields=["question_text"]),
    ), (
        "date information",
        dict(fields=["pub_date"]),
    )]
    inlines = [ChoiceInline]


admin.site.register(Question, QuestionAdmin)
