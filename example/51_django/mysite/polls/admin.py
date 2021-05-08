from django.contrib import admin
from .models import Question, Choice


class ChoiceInline(admin.TabularInline):
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
    list_display = (
        "question_text",
        "pub_date",
        "was_published_recently",
    )


admin.site.register(Question, QuestionAdmin)
