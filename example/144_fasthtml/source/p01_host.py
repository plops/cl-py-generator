#!/usr/bin/env python3
# pip install -U python-fasthtml
# https://youtu.be/evAb2x34Jqk?t=312
import datetime
from fasthtml.common import *
def render(comment):
    return Li(A(comment.comment, href=f"/comments/{comment.id}"))
app, rt, comments, Comment=fast_app("data/comments.db", id=int, comment=str, user=str, created_at=str, render=render, pk="id")
@rt("/")
def get():
    create_comment=Form(Input(id="username", name="username", placeholder="username"), Textarea(id="comment", name="comment", placeholder="comment"), Button("Add Comment"), hx_post="/comments", hx_target="#comments", hx_swap="afterbegin")
    comments_list=Ul(*comments(order_by="id DESC"), id="comments")
    return Div(comments_list, create_comment, cls="container")
@rt("/comments")
async def post(comment: Comment):
    comment.created_ad=datetime.datetime.now().isoformat()
    return comments.insert(comment)
serve()