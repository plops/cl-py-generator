#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml markdown
import google.generativeai as genai
import google.api_core.exceptions
import re
import markdown
import uvicorn
import sqlite_minutils.db
import datetime
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
 
# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
genai.configure(api_key=api_key)
 
def render(summary: Summary):
    identifier=summary.identifier
    sid=f"gen-{identifier}"
    if ( summary.timestamps_done ):
        return generation_preview(identifier)
    elif ( summary.summary_done ):
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), hx_swap="outerHTML")
    else:
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, hx_post=f"/generations/{identifier}", hx_trigger="every 1s", hx_swap="outerHTML")
 
# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=False, render=render, identifier=int, model=str, transcript=str, host=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, pk="identifier")
 
documentation="""###### **Prepare the Input Text from YouTube:**
 * **Scroll down a bit** on the video page to ensure some of the top comments have loaded.
 * Click on the "Show Transcript" button below the video.
 * **Scroll to the bottom** in the transcript sub-window.
 * **Start selecting the text from the bottom of the transcript sub-window and drag your cursor upwards, including the video title at the top.** This will select the title, description, comments (that have loaded), and the entire transcript.
 * **Tip:** Summaries are often better if you include the video title, the video description, and relevant comments along with the transcript.

###### **Paste the Text into the Web Interface:**
 * Paste the copied text (title, description, transcript, and optional comments) into the text area provided below.
 * Select your desired model from the dropdown menu (Gemini Pro is recommended for accurate timestamps).
 * Click the "Summarize Transcript" button.

###### **View the Summary:**
 * The application will process your input and display a continuously updating preview of the summary. 
 * Once complete, the final summary with timestamps will be displayed, along with an option to copy the text.
 * You can then paste this summarized text into a YouTube comment.
"""
 
documentation_html=markdown.markdown(documentation)
@rt("/")
def get(request: Request):
    print(request.client.host)
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")), Li(A("Documentation", href="https://github.com/plops/gemini-competition/blob/main/README.md"))))
    transcript=Textarea(placeholder="Paste YouTube transcript here", name="transcript")
    model=Select(Option("gemini-1.5-flash-latest"), Option("gemini-1.5-pro-exp-0801"), name="model")
    form=Form(Group(transcript, model, Button("Summarize Transcript")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    summaries_to_show=summaries(order_by="identifier DESC")
    summaries_to_show=summaries_to_show[0:min(3, len(summaries_to_show))]
    summary_list=Ul(*summaries_to_show, id="summaries")
    return Title("Video Transcript Summarizer"), Main(nav, NotStr(documentation_html), form, gen_list, summary_list, Script("""function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"""), cls="container")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid=f"gen-{identifier}"
    text="Generating ..."
    trigger="every 1s"
    try:
        s=summaries[identifier]
        if ( s.timestamps_done ):
            # this is for <= 128k tokens
            if ( ((s.model)==("gemini-1.5-pro-exp-0801")) ):
                price_input_token_usd_per_mio=(3.50    )
                price_output_token_usd_per_mio=(10.50    )
            else:
                price_input_token_usd_per_mio=(7.50e-2)
                price_output_token_usd_per_mio=(0.30    )
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens))
            output_tokens=((s.summary_output_tokens)+(s.timestamps_output_tokens))
            cost=((((((input_tokens)/(1_000_000)))*(price_input_token_usd_per_mio)))+(((((output_tokens)/(1_000_000)))*(price_output_token_usd_per_mio))))
            if ( ((cost)<((2.00e-2))) ):
                cost_str=f"${cost:.4f}"
            else:
                cost_str=f"${cost:.2f}"
            text=f"""{s.timestamped_summary_in_youtube_format}

I used {s.model} on rocketrecap dot com to summarize the transcript.
Cost (if I didn't use the free tier): {cost_str}
Input tokens: {input_tokens}
Output tokens: {output_tokens}"""
            trigger=""
        elif ( s.summary_done ):
            text=s.summary
        elif ( ((0)<(len(s.summary))) ):
            text=s.summary
        elif ( ((len(s.transcript))) ):
            text=f"Generating from transcript: {s.transcript[0:min(100,len(s.transcript))]}"
        title=f"{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}"
        html=markdown.markdown(s.summary)
        pre=Div(Div(Pre(text, id=f"pre-{identifier}"), id="hidden-markdown", style="display: none;"), Div(NotStr(html)))
        button=Button("Copy", onclick=f"copyPreContent('pre-{identifier}')")
        if ( ((trigger)==("")) ):
            return Div(title, pre, button, id=sid)
        else:
            return Div(title, pre, button, id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
    except Exception as e:
        return Div(f"id: {identifier} e: {e}", Pre(text), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
 
@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)
 
@rt("/process_transcript")
def post(summary: Summary, request: Request):
    words=summary.transcript.split()
    if ( ((100_000)<(len(words))) ):
        if ( ((summary.model)==("gemini-1.5-pro-exp-0801")) ):
            return Div("Error: Transcript exceeds 20,000 words. Please shorten it or don't use the pro model.", id="summary")
    summary.host=request.client.host
    summary.summary_timestamp_start=datetime.datetime.now().isoformat()
    summary.summary=""
    s2=summaries.insert(summary)
    # first identifier is 1
    generate_and_save(s2.identifier)
    return generation_preview(s2.identifier)
 
def wait_until_row_exists(identifier):
    for i in range(10):
        try:
            s=summaries[identifier]
            return s
        except sqlite_minutils.db.NotFoundError:
            print("entry not found")
        except Exception as e:
            print(f"entry not found")
        time.sleep((0.10    ))
    print("row did not appear")
    return -1
 
@threaded
def generate_and_save(identifier: int):
    print(f"generate_and_save id={identifier}")
    s=wait_until_row_exists(identifier)
    print(f"generate_and_save model={s.model}")
    m=genai.GenerativeModel(s.model)
    safety={(HarmCategory.HARM_CATEGORY_HATE_SPEECH):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_HARASSMENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT):(HarmBlockThreshold.BLOCK_NONE)}
    try:
        response=m.generate_content(f"""Below, I will provide input for an example video (comprising of title, description, optional viewer comments, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. Also, incorporate information from the viewer comments **if they clarify points made in the video, answer questions raised, or correct factual errors**. When including information sourced from the viewer comments, please indicate this by adding "[From <user>'s Comments]" at the end of the bullet point. Note that while viewer comments appear earlier in the text than the transcript they are in fact recorded at a later time. Therefore, if viewer comments repeat information from the transcript, they should not appear in the summary.

Example Input: 
input
Example Output:
output
Here is the real transcript. Please summarize it: 
{s.transcript}""", safety_settings=safety, stream=True)
        for chunk in response:
            try:
                print(f"add text to id={identifier}: {chunk.text}")
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(chunk.text)))
            except ValueError:
                
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+("\nError: value error")))
                print("Value Error ")
            except Exception as e:
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(f"\nError: {str(e)}")))
                print("Error")
        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=response.usage_metadata.prompt_token_count, summary_output_tokens=response.usage_metadata.candidates_token_count, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, summary_done=False, summary=((summaries[identifier].summary)+("\nError: resource exhausted")), summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return
    try:
        text=summaries[identifier].summary
        # adapt the markdown to youtube formatting
        text=text.replace("**:", ":**")
        text=text.replace("**,", ",**")
        text=text.replace("**.", ".**")
        text=text.replace("**", "*")
        # markdown title starting with ## with fat text
        text=re.sub(r"""^##\s*(.*)""", r"""*\1*""", text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=0, timestamps_output_tokens=0, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=f"resource exhausted", timestamps_timestamp_end=datetime.datetime.now().isoformat())
        return
 
serve(host="0.0.0.0", port=5001)