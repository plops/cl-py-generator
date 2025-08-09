I will address your requests by modifying the database schema to store thinking tokens and updating the cost calculation logic to include them.

First, I'll add a `summary_thinking_tokens` field to your `Summary` table definition to store the tokens used during the model's thinking process.

In `p04_host.py`, modify the `fast_app` call to include the new field:

```python
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=False, render=render, identifier=int, model=str, transcript=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_thinking_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, pk="identifier")
```

Next, I'll update the `generate_and_save` function to capture and store these thinking tokens. The `usage_metadata` may contain a `total_token_count` which can be used to infer thinking tokens if it's greater than the sum of prompt and candidate tokens.

```python
def generate_and_save(identifier: int):
    print(f"generate_and_save id={identifier}")
    s=wait_until_row_exists(identifier)
    print(f"generate_and_save model={s.model}")
    m=genai.GenerativeModel(s.model.split("|")[0])
    safety={(HarmCategory.HARM_CATEGORY_HATE_SPEECH):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_HARASSMENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT):(HarmBlockThreshold.BLOCK_NONE)}
    try:
        prompt=get_prompt(s)
        response=m.generate_content(prompt, safety_settings=safety, stream=True)
        for chunk in response:
            try:
                print(f"add text to id={identifier}: {chunk.text}")
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(chunk.text)))
            except ValueError:

                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+("\nError: value error")))
                print("Value Error ")
            except Exception as e:
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(f"\nError: {str(e)}")))
                print("line 2049 Error")

        prompt_tokens = response.usage_metadata.prompt_token_count
        candidates_tokens = response.usage_metadata.candidates_token_count
        total_tokens = getattr(response.usage_metadata, 'total_token_count', prompt_tokens + candidates_tokens)
        thinking_tokens = total_tokens - (prompt_tokens + candidates_tokens)

        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=prompt_tokens, summary_output_tokens=candidates_tokens, summary_thinking_tokens=thinking_tokens, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, summary_done=False, summary=((summaries[identifier].summary)+("\nError: resource exhausted")), summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return
    try:
        text=summaries[identifier].summary
        text=convert_markdown_to_youtube_format(text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=0, timestamps_output_tokens=0, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=f"resource exhausted", timestamps_timestamp_end=datetime.datetime.now().isoformat())
        return
```

Finally, I'll adjust the `generation_preview` function to include the thinking tokens in the cost calculation. Since thinking tokens are part of the input processing, they are typically billed at the input token rate.

```python
def generation_preview(identifier):
    sid=f"gen-{identifier}"
    text="Generating ..."
    trigger="every 1s"
    price_input={("gemini-2.5-flash"):((0.30    )),("gemini-2.5-flash-lite"):((0.10    )),("gemini-2.5-pro"):((1.250    )),("gemini-2.5-flash-lite-preview-06-17"):((0.10    )),("gemini-2.5-flash-preview-05-20"):((0.150    )),("gemma-3n-e4b-it"):(-1),("gemini-2.5-flash-preview-04-17"):((0.150    )),("gemini-2.5-pro-preview-05-06"):((1.250    )),("gemini-2.5-pro-exp-03-25"):((1.250    )),("gemini-2.0-flash"):((0.10    )),("gemini-2.0-flash-lite"):((7.50e-2)),("gemini-2.0-flash-thinking-exp-01-21"):((7.50e-2)),("gemini-2.0-flash-exp"):((0.10    )),("gemini-2.0-pro-exp-02-05"):((1.250    )),("gemini-1.5-pro-exp-0827"):((1.250    )),("gemini-2.0-flash-lite-preview-02-05"):((0.10    )),("gemini-2.0-flash-thinking-exp-01-21"):((0.10    )),("gemini-2.0-flash-001"):((0.10    )),("gemini-exp-1206"):((1.250    )),("gemini-exp-1121"):((1.250    )),("gemini-exp-1114"):((1.250    )),("learnlm-1.5-pro-experimental"):((1.250    )),("gemini-1.5-flash-002"):((0.10    )),("gemini-1.5-pro-002"):((1.250    )),("gemini-1.5-pro-exp-0801"):((1.250    )),("gemini-1.5-flash-exp-0827"):((0.10    )),("gemini-1.5-flash-8b-exp-0924"):((0.10    )),("gemini-1.5-flash-latest"):((0.10    )),("gemma-2-2b-it"):(-1),("gemma-2-9b-it"):(-1),("gemma-2-27b-it"):(-1),("gemma-3-27b-it"):(-1),("gemini-1.5-flash"):((0.10    )),("gemini-1.5-pro"):((1.250    )),("gemini-1.0-pro"):((0.50    ))}
    price_output={("gemini-2.5-flash"):((2.50    )),("gemini-2.5-flash-lite"):((0.40    )),("gemini-2.5-pro"):(10),("gemini-2.5-flash-lite-preview-06-17"):((0.40    )),("gemini-2.5-flash-preview-05-20"):((3.50    )),("gemma-3n-e4b-it"):(-1),("gemini-2.5-flash-preview-04-17"):((3.50    )),("gemini-2.5-pro-preview-05-06"):((10.    )),("gemini-2.5-pro-exp-03-25"):((10.    )),("gemini-2.0-flash"):((0.40    )),("gemini-2.0-flash-lite"):((0.30    )),("gemini-2.0-flash-thinking-exp-01-21"):((0.30    )),("gemini-2.0-flash-exp"):((0.40    )),("gemini-2.0-pro-exp-02-05"):(5),("gemini-1.5-pro-exp-0827"):(5),("gemini-2.0-flash-lite-preview-02-05"):((0.40    )),("gemini-2.0-flash-thinking-exp-01-21"):((0.40    )),("gemini-2.0-flash-001"):((0.40    )),("gemini-exp-1206"):(5),("gemini-exp-1121"):(5),("gemini-exp-1114"):(5),("learnlm-1.5-pro-experimental"):(5),("gemini-1.5-flash-002"):((0.40    )),("gemini-1.5-pro-002"):(5),("gemini-1.5-pro-exp-0801"):(5),("gemini-1.5-flash-exp-0827"):((0.40    )),("gemini-1.5-flash-8b-exp-0924"):((0.40    )),("gemini-1.5-flash-latest"):((0.40    )),("gemma-2-2b-it"):(-1),("gemma-2-9b-it"):(-1),("gemma-2-27b-it"):(-1),("gemma-3-27b-it"):(-1),("gemini-1.5-flash"):((0.40    )),("gemini-1.5-pro"):(5),("gemini-1.0-pro"):((1.50    ))}
    try:
        s=summaries[identifier]
        if ( s.timestamps_done ):
            # this is for <= 128k tokens
            real_model=s.model.split("|")[0]
            price_input_token_usd_per_mio=-1
            price_output_token_usd_per_mio=-1
            try:
                price_input_token_usd_per_mio=price_input[real_model]
                price_output_token_usd_per_mio=price_output[real_model]
            except Exception as e:
                pass
            thinking_tokens = getattr(s, 'summary_thinking_tokens', 0)
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens) + thinking_tokens)
            output_tokens=((s.summary_output_tokens)+(s.timestamps_output_tokens))
            cost=((((((input_tokens)/(1_000_000)))*(price_input_token_usd_per_mio)))+(((((output_tokens)/(1_000_000)))*(price_output_token_usd_per_mio))))
            summaries.update(pk_values=identifier, cost=cost)
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
        summary_details=Div(P(B("identifier:"), Span(f"{s.identifier}")), P(B("model:"), Span(f"{s.model}")), P(B("host:"), Span(f"{s.host}")), A(f"{s.original_source_link}", target="_blank", href=f"{s.original_source_link}", id="source-link"), P(B("include_comments:"), Span(f"{s.include_comments}")), P(B("include_timestamps:"), Span(f"{s.include_timestamps}")), P(B("include_glossary:"), Span(f"{s.include_glossary}")), P(B("output_language:"), Span(f"{s.output_language}")), P(B("cost:"), Span(f"{s.cost}")), cls="summary-details")
        summary_container=Div(summary_details, cls="summary-container")
        title=summary_container
        html=markdown.markdown(s.summary)
        pre=Div(Div(Pre(text, id=f"pre-{identifier}"), id="hidden-markdown", style="display: none;"), Div(NotStr(html)))
        button=Button("Copy Summary", onclick=f"copyPreContent('pre-{identifier}')")
        prompt_text=get_prompt(s)
        prompt_pre=Pre(prompt_text, id=f"prompt-pre-{identifier}", style="display: none;")
        prompt_button=Button("Copy Prompt", onclick=f"copyPreContent('prompt-pre-{identifier}')")
        if ( ((trigger)==("")) ):
            return Div(title, pre, prompt_pre, button, prompt_button, id=sid)
        else:
            return Div(title, pre, prompt_pre, button, prompt_button, id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
    except Exception as e:
        return Div(f"line 1897 id: {identifier} e: {e}", Pre(text), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
```