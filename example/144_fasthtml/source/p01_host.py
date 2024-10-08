#!/usr/bin/env python3
# pip install -U python-fasthtml
# https://youtu.be/evAb2x34Jqk?t=312
import datetime
from fasthtml.common import *
def render(comment):
    return Li(A(comment.comment, href=f"/comments/{comment.id}"), f"by {comment.user} {comment.created_at}")
app, rt, comments, Comment=fast_app("data/comments.db", id=int, comment=str, user=str, created_at=str, render=render, pk="id")
@rt("/")
def get():
    nav=Nav(Ul(Li(Strong("Acme Corp"))), Ul(Li(A("About", href="#")), Li(A("Services", href="#")), Li(A("Products", href="#"))))
    create_comment=Form(Input(id="username", name="user", placeholder="username"), Textarea(id="comment", name="comment", placeholder="comment"), Button("Add Comment"), hx_post="/comments", hx_target="#comments", hx_swap="afterbegin")
    comments_list=Ul(*comments(order_by="id DESC"), id="comments")
    return Div(nav, comments_list, create_comment, cls="container")
@rt("/comments")
async def post(comment: Comment):
    comment.created_at=datetime.datetime.now().isoformat()
    return comments.insert(comment)
serve()