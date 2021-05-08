from django.contrib import admin
from .models import Question


class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [(
        None,
        dict(fields=["question_text"]),
    ), (
        "date information",
        dict(fields=["pub_date"]),
    )]


admin.site.register(Question, QuestionAdmin)
