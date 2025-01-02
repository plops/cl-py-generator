#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml markdown
# micromamba install python-fasthtml markdown yt-dlp; pip install  webvtt-py
import google.generativeai as genai
import os
import google.api_core.exceptions
import markdown
import sqlite_minutils.db
import datetime
import subprocess
import webvtt
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
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=False, render=render, identifier=int, model=str, transcript=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, pk="identifier")
 
documentation="""###### To use the YouTube summarizer:

1. **Copy the YouTube video link.**
2. **Paste the link into the provided input field.**
3. **Click the 'Summarize' button.** The summary with timestamps will be generated.

"""
 
def get_transcript(url):
    # Call yt-dlp to download the subtitles
    pattern=r"""^https://(www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}$"""
    if ( not(re.match(pattern, url)) ):
        print("Error: Invalid youtube url")
        return ""
    sub_file="/dev/shm/o"
    sub_file_="/dev/shm/o.en.vtt"
    cmds=["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs", "--cookies-from-browser", "firefox", "--sub-lang", "en", "-o", sub_file, url]
    print(" ".join(cmds))
    subprocess.run(cmds)
    ostr=""
    try:
        for c in webvtt.read(sub_file_):
            # we don't need sub-second time resolution. trim it away
            start=c.start.split(".")[0]
            # remove internal newlines within a caption
            cap=c.text.strip().replace("\n", " ")
            # write <start> <c.text> into each line of ostr
            ostr += f"{start} {cap}\n"
        os.remove(sub_file_)
    except FileNotFoundError:
        print("Error: Subtitle file not found")
    except Exception as e:
        print("Error: problem when processing subtitle file")
    return ostr
 
documentation_html=markdown.markdown(documentation)
@rt("/")
def get(request: Request):
    print(request.client.host)
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")), Li(A("Documentation", href="https://github.com/plops/gemini-competition/blob/main/README.md"))))
    transcript=Textarea(placeholder="(Optional) Paste YouTube transcript here", style="height: 300px; width=60%; visibility: hidden;", name="transcript")
    model=Div(Select(Option("gemini-1.5-pro-exp-0827"), Option("gemini-2.0-flash-exp"), Option("gemini-exp-1206"), Option("gemini-exp-1121"), Option("gemini-exp-1114"), Option("learnlm-1.5-pro-experimental"), Option("gemini-1.5-flash-002"), Option("gemini-1.5-pro-002"), Option("gemini-1.5-pro-exp-0801"), Option("gemini-1.5-flash-exp-0827"), Option("gemini-1.5-flash-8b-exp-0924"), Option("gemini-1.5-flash-latest"), Option("gemma-2-2b-it"), Option("gemma-2-9b-it"), Option("gemma-2-27b-it"), Option("gemini-1.5-flash"), Option("gemini-1.5-pro"), Option("gemini-1.0-pro"), style="width: 100%;", name="model"), style="display: flex; align-items: center; width: 100%;")
    form=Form(Group(Div(Textarea(placeholder="Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)", name="original_source_link"), transcript, model, Div(Label("Output Language", _for="output_language"), Select(Option("en"), Option("de"), Option("fr"), Option("ch"), Option("nl"), Option("pt"), Option("cz"), Option("it"), Option("jp"), Option("ar"), style="width: 100%;", name="output_language", id="output_language"), style="display: flex; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_comments", name="include_comments", checked=False), Label("Include User Comments", _for="include_comments"), style="display: flex; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_timestamps", name="include_timestamps", checked=True), Label("Include Timestamps", _for="include_timestamps"), style="display: flex; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_glossary", name="include_glossary", checked=False), Label("Include Glossary", _for="include_glossary"), style="display: flex; align-items: center; width: 100%;"), Button("Summarize Transcript"), style="display: flex; flex-direction:column;")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    summaries_to_show=summaries(order_by="identifier DESC")
    summaries_to_show=summaries_to_show[0:min(13, len(summaries_to_show))]
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
            if ( s.model.startswith("gemini-1.5-pro") ):
                price_input_token_usd_per_mio=(1.250    )
                price_output_token_usd_per_mio=(5.0    )
            else:
                if ( s.model.startswith("gemini-1.5-flash") ):
                    price_input_token_usd_per_mio=(7.50e-2)
                    price_output_token_usd_per_mio=(0.30    )
                else:
                    if ( s.model.startswith("gemini-1.0-pro") ):
                        price_input_token_usd_per_mio=(0.50    )
                        price_output_token_usd_per_mio=(1.50    )
                    else:
                        price_input_token_usd_per_mio=-1
                        price_output_token_usd_per_mio=-1
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens))
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
        title=f"identifier: {s.identifier} model: {s.model} host: {s.host} original_source_link: {s.original_source_link} include_comments: {s.include_comments} include_timestamps: {s.include_timestamps} include_glossary: {s.include_glossary} output_language: {s.output_language} cost: {s.cost}"
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
    if ( ((0)==(len(summary.transcript))) ):
        # No transcript given, try to download from URL
        summary.transcript=get_transcript(summary.original_source_link)
    words=summary.transcript.split()
    if ( ((len(words))<(30)) ):
        return Div("Error: Transcript is too short. No summary necessary", id="summary")
    if ( ((100_000)<(len(words))) ):
        if ( summary.model.startswith("gemini-1.5-pro") ):
            return Div("Error: Transcript exceeds 20,000 words. Please shorten it or don't use the pro model.", id="summary")
    summary.host=request.client.host
    summary.summary_timestamp_start=datetime.datetime.now().isoformat()
    print(f"link: {summary.original_source_link}")
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
            print(f"unknown exception {e}")
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
        prompt=f"""Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
Fluidigm Polaris Part 2- illuminator and camera
mikeselectricstuff
131K subscribers
Subscribed
369
Share
Download
Clip
Save
5,857 views Aug 26, 2024
Fluidigm Polaris part 1 : • Fluidigm Polaris (Part 1) - Biotech g...
Ebay listings: https://www.ebay.co.uk/usr/mikeselect...
Merch https://mikeselectricstuff.creator-sp...
Transcript
Follow along using the transcript.
Show transcript
mikeselectricstuff
131K subscribers
Videos
About
Support on Patreon
40 Comments
@robertwatsonbath
6 hours ago
Thanks Mike. Ooof! - with the level of bodgery going on around 15:48 I think shame would have made me do a board re spin, out of my own pocket if I had to.
1
Reply
@Muonium1
9 hours ago
The green LED looks different from the others and uses phosphor conversion because of the "green gap" problem where green InGaN emitters suffer efficiency droop at high currents. Phosphide based emitters don't start becoming efficient until around 600nm so also can't be used for high power green emitters. See the paper and plot by Matthias Auf der Maur in his 2015 paper on alloy fluctuations in InGaN as the cause of reduced external quantum efficiency at longer (green) wavelengths.
4
Reply
1 reply
@tafsirnahian669
10 hours ago (edited)
Can this be used as an astrophotography camera?
Reply
mikeselectricstuff
·
1 reply
@mikeselectricstuff
6 hours ago
Yes, but may need a shutter to avoid light during readout
Reply
@2010craggy
11 hours ago
Narrowband filters we use in Astronomy (Astrophotography) are sided- they work best passing light in one direction so I guess the arrows on the filter frames indicate which way round to install them in the filter wheel.
1
Reply
@vitukz
12 hours ago
A mate with Channel @extractions&ire could use it
2
Reply
@RobertGallop
19 hours ago
That LED module says it can go up to 28 amps!!! 21 amps for 100%. You should see what it does at 20 amps!
Reply
@Prophes0r
19 hours ago
I had an "Oh SHIT!" moment when I realized that the weird trapezoidal shape of that light guide was for keystone correction of the light source.
Very clever.
6
Reply
@OneBiOzZ
20 hours ago
given the cost of the CCD you think they could have run another PCB for it
9
Reply
@tekvax01
21 hours ago
$20 thousand dollars per minute of run time!
1
Reply
@tekvax01
22 hours ago
"We spared no expense!" John Hammond Jurassic Park.
*(that's why this thing costs the same as a 50-seat Greyhound Bus coach!)
Reply
@florianf4257
22 hours ago
The smearing on the image could be due to the fact that you don't use a shutter, so you see brighter stripes under bright areas of the image as you still iluminate these pixels while the sensor data ist shifted out towards the top. I experienced this effect back at university with a LN-Cooled CCD for Spectroscopy. The stripes disapeared as soon as you used the shutter instead of disabling it in the open position (but fokussing at 100ms integration time and continuous readout with a focal plane shutter isn't much fun).
12
Reply
mikeselectricstuff
·
1 reply
@mikeselectricstuff
12 hours ago
I didn't think of that, but makes sense
2
Reply
@douro20
22 hours ago (edited)
The red LED reminds me of one from Roithner Lasertechnik. I have a Symbol 2D scanner which uses two very bright LEDs from that company, one red and one red-orange. The red-orange is behind a lens which focuses it into an extremely narrow beam.
1
Reply
@RicoElectrico
23 hours ago
PFG is Pulse Flush Gate according to the datasheet.
Reply
@dcallan812
23 hours ago
Very interesting. 2x
Reply
@littleboot_
1 day ago
Cool interesting device
Reply
@dav1dbone
1 day ago
I've stripped large projectors, looks similar, wonder if some of those castings are a magnesium alloy?
Reply
@kevywevvy8833
1 day ago
ironic that some of those Phlatlight modules are used in some of the cheapest disco lights.
1
Reply
1 reply
@bill6255
1 day ago
Great vid - gets right into subject in title, its packed with information, wraps up quickly. Should get a YT award! imho
3
Reply
@JAKOB1977
1 day ago (edited)
The whole sensor module incl. a 5 grand 50mpix sensor for 49 £.. highest bid atm
Though also a limited CCD sensor, but for the right buyer its a steal at these relative low sums.
Architecture Full Frame CCD (Square Pixels)
Total Number of Pixels 8304 (H) × 6220 (V) = 51.6 Mp
Number of Effective Pixels 8208 (H) × 6164 (V) = 50.5 Mp
Number of Active Pixels 8176 (H) × 6132 (V) = 50.1 Mp
Pixel Size 6.0 m (H) × 6.0 m (V)
Active Image Size 49.1 mm (H) × 36.8 mm (V)
61.3 mm (Diagonal),
645 1.1x Optical Format
Aspect Ratio 4:3
Horizontal Outputs 4
Saturation Signal 40.3 ke−
Output Sensitivity 31 V/e−
Quantum Efficiency
KAF−50100−CAA
KAF−50100−AAA
KAF−50100−ABA (with Lens)
22%, 22%, 16% (Peak R, G, B)
25%
62%
Read Noise (f = 18 MHz) 12.5 e−
Dark Signal (T = 60°C) 42 pA/cm2
Dark Current Doubling Temperature 5.7°C
Dynamic Range (f = 18 MHz) 70.2 dB
Estimated Linear Dynamic Range
(f = 18 MHz)
69.3 dB
Charge Transfer Efficiency
Horizontal
Vertical
0.999995
0.999999
Blooming Protection
(4 ms Exposure Time)
800X Saturation Exposure
Maximum Date Rate 18 MHz
Package Ceramic PGA
Cover Glass MAR Coated, 2 Sides or
Clear Glass
Features
• TRUESENSE Transparent Gate Electrode
for High Sensitivity
• Ultra-High Resolution
• Board Dynamic Range
• Low Noise Architecture
• Large Active Imaging Area
Applications
• Digitization
• Mapping/Aerial
• Photography
• Scientific
Thx for the tear down Mike, always a joy
Reply
@martinalooksatthings
1 day ago
15:49 that is some great bodging on of caps, they really didn't want to respin that PCB huh
8
Reply
@RhythmGamer
1 day ago
Was depressed today and then a new mike video dropped and now I’m genuinely happy to get my tear down fix
1
Reply
@dine9093
1 day ago (edited)
Did you transfrom into Mr Blobby for a moment there?
2
Reply
@NickNorton
1 day ago
Thanks Mike. Your videos are always interesting.
5
Reply
@KeritechElectronics
1 day ago
Heavy optics indeed... Spare no expense, cost no object. Splendid build quality. The CCD is a thing of beauty!
1
Reply
@YSoreil
1 day ago
The pricing on that sensor is about right, I looked in to these many years ago when they were still in production since it's the only large sensor you could actually buy. Really cool to see one in the wild.
2
Reply
@snik2pl
1 day ago
That leds look like from led projector
Reply
@vincei4252
1 day ago
TDI = Time Domain Integration ?
1
Reply
@wolpumba4099
1 day ago (edited)
Maybe the camera should not be illuminated during readout.
From the datasheet of the sensor (Onsemi): saturation 40300 electrons, read noise 12.5 electrons per pixel @ 18MHz (quite bad). quantum efficiency 62% (if it has micro lenses), frame rate 1 Hz. lateral overflow drain to prevent blooming protects against 800x (factor increases linearly with exposure time) saturation exposure (32e6 electrons per pixel at 4ms exposure time), microlens has +/- 20 degree acceptance angle
i guess it would be good for astrophotography
4
Reply
@txm100
1 day ago (edited)
Babe wake up a new mikeselectricstuff has dropped!
9
Reply
@vincei4252
1 day ago
That looks like a finger-lakes filter wheel, however, for astronomy they'd never use such a large stepper.
1
Reply
@MRooodddvvv
1 day ago
yaaaaay ! more overcomplicated optical stuff !
4
Reply
1 reply
@NoPegs
1 day ago
He lives!
11
Reply
1 reply
Transcript
0:00
so I've stripped all the bits of the
0:01
optical system so basically we've got
0:03
the uh the camera
0:05
itself which is mounted on this uh very
0:09
complex
0:10
adjustment thing which obviously to set
0:13
you the various tilt and uh alignment
0:15
stuff then there's two of these massive
0:18
lenses I've taken one of these apart I
0:20
think there's something like about eight
0:22
or nine Optical elements in here these
0:25
don't seem to do a great deal in terms
0:26
of electr magnification they're obiously
0:28
just about getting the image to where it
0:29
uh where it needs to be just so that
0:33
goes like that then this Optical block I
0:36
originally thought this was made of some
0:37
s crazy heavy material but it's just
0:39
really the sum of all these Optical bits
0:41
are just ridiculously heavy those lenses
0:43
are about 4 kilos each and then there's
0:45
this very heavy very solid um piece that
0:47
goes in the middle and this is so this
0:49
is the filter wheel assembly with a
0:51
hilariously oversized steper
0:53
motor driving this wheel with these very
0:57
large narrow band filters so we've got
1:00
various different shades of uh
1:03
filters there five Al together that
1:06
one's actually just showing up a silver
1:07
that's actually a a red but fairly low
1:10
transmission orangey red blue green
1:15
there's an excess cover on this side so
1:16
the filters can be accessed and changed
1:19
without taking anything else apart even
1:21
this is like ridiculous it's like solid
1:23
aluminium this is just basically a cover
1:25
the actual wavelengths of these are um
1:27
488 525 570 630 and 700 NM not sure what
1:32
the suffix on that perhaps that's the uh
1:34
the width of the spectral line say these
1:37
are very narrow band filters most of
1:39
them are you very little light through
1:41
so it's still very tight narrow band to
1:43
match the um fluoresence of the dies
1:45
they're using in the biochemical process
1:48
and obviously to reject the light that's
1:49
being fired at it from that Illuminator
1:51
box and then there's a there's a second
1:53
one of these lenses then the actual sort
1:55
of samples below that so uh very serious
1:58
amount of very uh chunky heavy Optics
2:01
okay let's take a look at this light
2:02
source made by company Lumen Dynamics
2:04
who are now part of
2:06
excelitas self-contained unit power
2:08
connector USB and this which one of the
2:11
Cable Bundle said was a TTL interface
2:14
USB wasn't used in uh the fluid
2:17
application output here and I think this
2:19
is an input for um light feedback I
2:21
don't if it's regulated or just a measur
2:23
measurement facility and the uh fiber
2:27
assembly
2:29
Square Inlet there and then there's two
2:32
outputs which have uh lens assemblies
2:35
and this small one which goes back into
2:37
that small Port just Loops out of here
2:40
straight back in So on this side we've
2:42
got the electronics which look pretty
2:44
straightforward we've got a bit of power
2:45
supply stuff over here and we've got
2:48
separate drivers for each wavelength now
2:50
interesting this is clearly been very
2:52
specifically made for this application
2:54
you I was half expecting like say some
2:56
generic drivers that could be used for a
2:58
number of different things but actually
3:00
literally specified the exact wavelength
3:02
on the PCB there is provision here for
3:04
385 NM which isn't populated but this is
3:07
clearly been designed very specifically
3:09
so these four drivers look the same but
3:10
then there's two higher power ones for
3:12
575 and
3:14
520 a slightly bigger heat sink on this
3:16
575 section there a p 24 which is
3:20
providing USB interface USB isolator the
3:23
USB interface just presents as a comport
3:26
I did have a quick look but I didn't
3:27
actually get anything sensible um I did
3:29
dump the Pi code out and there's a few
3:31
you a few sort of commands that you
3:32
could see in text but I didn't actually
3:34
manage to get it working properly I
3:36
found some software for related version
3:38
but it didn't seem to want to talk to it
3:39
but um I say that wasn't used for the
3:41
original application it might be quite
3:42
interesting to get try and get the Run
3:44
hours count out of it and the TTL
3:46
interface looks fairly straightforward
3:48
we've got positions for six opto
3:50
isolators but only five five are
3:52
installed so that corresponds with the
3:54
unused thing so I think this hopefully
3:56
should be as simple as just providing a
3:57
ttrl signal for each color to uh enable
4:00
it a big heat sink here which is there I
4:03
think there's like a big S of metal
4:04
plate through the middle of this that
4:05
all the leads are mounted on the other
4:07
side so this is heat sinking it with a
4:09
air flow from a uh just a fan in here
4:13
obviously don't have the air flow
4:14
anywhere near the Optics so conduction
4:17
cool through to this plate that's then
4:18
uh air cooled got some pots which are
4:21
presumably power
4:22
adjustments okay let's take a look at
4:24
the other side which is uh much more
4:27
interesting see we've got some uh very
4:31
uh neatly Twisted cable assemblies there
4:35
a bunch of leads so we've got one here
4:37
475 up here 430 NM 630 575 and 520
4:44
filters and dcro mirrors a quick way to
4:48
see what's white is if we just shine
4:49
some white light through
4:51
here not sure how it is is to see on the
4:54
camera but shining white light we do
4:55
actually get a bit of red a bit of blue
4:57
some yellow here so the obstacle path
5:00
575 it goes sort of here bounces off
5:03
this mirror and goes out the 520 goes
5:07
sort of down here across here and up
5:09
there 630 goes basically straight
5:13
through
5:15
430 goes across there down there along
5:17
there and the 475 goes down here and
5:20
left this is the light sensing thing
5:22
think here there's just a um I think
5:24
there a photo diode or other sensor
5:26
haven't actually taken that off and
5:28
everything's fixed down to this chunk of
5:31
aluminium which acts as the heat
5:32
spreader that then conducts the heat to
5:33
the back side for the heat
5:35
sink and the actual lead packages all
5:38
look fairly similar except for this one
5:41
on the 575 which looks quite a bit more
5:44
substantial big spay
5:46
Terminals and the interface for this
5:48
turned out to be extremely simple it's
5:50
literally a 5V TTL level to enable each
5:54
color doesn't seem to be any tensity
5:56
control but there are some additional
5:58
pins on that connector that weren't used
5:59
in the through time thing so maybe
6:01
there's some extra lines that control
6:02
that I couldn't find any data on this uh
6:05
unit and the um their current product
6:07
range is quite significantly different
6:09
so we've got the uh blue these
6:13
might may well be saturating the camera
6:16
so they might look a bit weird so that's
6:17
the 430
6:18
blue the 575
6:24
yellow uh
6:26
475 light blue
6:29
the uh 520
6:31
green and the uh 630 red now one
6:36
interesting thing I noticed for the
6:39
575 it's actually it's actually using a
6:42
white lead and then filtering it rather
6:44
than using all the other ones are using
6:46
leads which are the fundamental colors
6:47
but uh this is actually doing white and
6:50
it's a combination of this filter and
6:52
the dichroic mirrors that are turning to
6:55
Yellow if we take the filter out and a
6:57
lot of the a lot of the um blue content
7:00
is going this way the red is going
7:02
straight through these two mirrors so
7:05
this is clearly not reflecting much of
7:08
that so we end up with the yellow coming
7:10
out of uh out of there which is a fairly
7:14
light yellow color which you don't
7:16
really see from high intensity leads so
7:19
that's clearly why they've used the
7:20
white to uh do this power consumption of
7:23
the white is pretty high so going up to
7:25
about 2 and 1 half amps on that color
7:27
whereas most of the other colors are
7:28
only drawing half an amp or so at 24
7:30
volts the uh the green is up to about
7:32
1.2 but say this thing is uh much
7:35
brighter and if you actually run all the
7:38
colors at the same time you get a fairly
7:41
reasonable um looking white coming out
7:43
of it and one thing you might just be
7:45
out to notice is there is some sort
7:46
color banding around here that's not
7:49
getting uh everything s completely
7:51
concentric and I think that's where this
7:53
fiber optic thing comes
7:58
in I'll
8:00
get a couple of Fairly accurately shaped
8:04
very sort of uniform color and looking
8:06
at What's um inside here we've basically
8:09
just got this Square Rod so this is
8:12
clearly yeah the lights just bouncing
8:13
off all the all the various sides to um
8:16
get a nice uniform illumination uh this
8:19
back bit looks like it's all potted so
8:21
nothing I really do to get in there I
8:24
think this is fiber so I have come
8:26
across um cables like this which are
8:27
liquid fill but just looking through the
8:30
end of this it's probably a bit hard to
8:31
see it does look like there fiber ends
8:34
going going on there and so there's this
8:36
feedback thing which is just obviously
8:39
compensating for the any light losses
8:41
through here to get an accurate
8:43
representation of uh the light that's
8:45
been launched out of these two
8:47
fibers and you see uh
8:49
these have got this sort of trapezium
8:54
shape light guides again it's like a
8:56
sort of acrylic or glass light guide
9:00
guess projected just to make the right
9:03
rectangular
9:04
shape and look at this Center assembly
9:07
um the light output doesn't uh change
9:10
whether you feed this in or not so it's
9:11
clear not doing any internal Clos Loop
9:14
control obviously there may well be some
9:16
facility for it to do that but it's not
9:17
being used in this
9:19
application and so this output just
9:21
produces a voltage on the uh outle
9:24
connector proportional to the amount of
9:26
light that's present so there's a little
9:28
diffuser in the back there
9:30
and then there's just some kind of uh
9:33
Optical sensor looks like a
9:35
chip looking at the lead it's a very
9:37
small package on the PCB with this lens
9:40
assembly over the top and these look
9:43
like they're actually on a copper
9:44
Metalized PCB for maximum thermal
9:47
performance and yeah it's a very small
9:49
package looks like it's a ceramic
9:51
package and there's a thermister there
9:53
for temperature monitoring this is the
9:56
475 blue one this is the 520 need to
9:59
Green which is uh rather different OB
10:02
it's a much bigger D with lots of bond
10:04
wise but also this looks like it's using
10:05
a phosphor if I shine a blue light at it
10:08
lights up green so this is actually a
10:10
phosphor conversion green lead which
10:12
I've I've come across before they want
10:15
that specific wavelength so they may be
10:17
easier to tune a phosphor than tune the
10:20
um semiconductor material to get the uh
10:23
right right wavelength from the lead
10:24
directly uh red 630 similar size to the
10:28
blue one or does seem to have a uh a
10:31
lens on top of it there is a sort of red
10:33
coloring to
10:35
the die but that doesn't appear to be
10:38
fluorescent as far as I can
10:39
tell and the white one again a little
10:41
bit different sort of much higher
10:43
current
10:46
connectors a makeer name on that
10:48
connector flot light not sure if that's
10:52
the connector or the lead
10:54
itself and obviously with the phosphor
10:56
and I'd imagine that phosphor may well
10:58
be tuned to get the maximum to the uh 5
11:01
cenm and actually this white one looks
11:04
like a St fairly standard product I just
11:06
found it in Mouse made by luminous
11:09
devices in fact actually I think all
11:11
these are based on various luminous
11:13
devices modules and they're you take
11:17
looks like they taking the nearest
11:18
wavelength and then just using these
11:19
filters to clean it up to get a precise
11:22
uh spectral line out of it so quite a
11:25
nice neat and um extreme
11:30
bright light source uh sure I've got any
11:33
particular use for it so I think this
11:35
might end up on
11:36
eBay but uh very pretty to look out and
11:40
without the uh risk of burning your eyes
11:43
out like you do with lasers so I thought
11:45
it would be interesting to try and
11:46
figure out the runtime of this things
11:48
like this we usually keep some sort
11:49
record of runtime cuz leads degrade over
11:51
time I couldn't get any software to work
11:52
through the USB face but then had a
11:54
thought probably going to be writing the
11:55
runtime periodically to the e s prom so
11:58
I just just scope up that and noticed it
12:00
was doing right every 5 minutes so I
12:02
just ran it for a while periodically
12:04
reading the E squ I just held the pick
12:05
in in reset and um put clip over to read
12:07
the square prom and found it was writing
12:10
one location per color every 5 minutes
12:12
so if one color was on it would write
12:14
that location every 5 minutes and just
12:16
increment it by one so after doing a few
12:18
tests with different colors of different
12:19
time periods it looked extremely
12:21
straightforward it's like a four bite
12:22
count for each color looking at the
12:24
original data that was in it all the
12:26
colors apart from Green were reading
12:28
zero and the green was reading four
12:30
indicating a total 20 minutes run time
12:32
ever if it was turned on run for a short
12:34
time then turned off that might not have
12:36
been counted but even so indicates this
12:37
thing wasn't used a great deal the whole
12:40
s process of doing a run can be several
12:42
hours but it'll only be doing probably
12:43
the Imaging at the end of that so you
12:46
wouldn't expect to be running for a long
12:47
time but say a single color for 20
12:50
minutes over its whole lifetime does
12:52
seem a little bit on the low side okay
12:55
let's look at the camera un fortunately
12:57
I managed to not record any sound when I
12:58
did this it's also a couple of months
13:00
ago so there's going to be a few details
13:02
that I've forgotten so I'm just going to
13:04
dub this over the original footage so um
13:07
take the lid off see this massive great
13:10
heat sink so this is a pel cool camera
13:12
we've got this blower fan producing a
13:14
fair amount of air flow through
13:16
it the connector here there's the ccds
13:19
mounted on the board on the
13:24
right this unplugs so we've got a bit of
13:27
power supply stuff on here
13:29
USB interface I think that's the Cyprus
13:32
microcontroller High speeded USB
13:34
interface there's a zyink spon fpga some
13:40
RAM and there's a couple of ATD
13:42
converters can't quite read what those
13:45
those are but anal
13:47
devices um little bit of bodgery around
13:51
here extra decoupling obviously they
13:53
have having some noise issues this is
13:55
around the ram chip quite a lot of extra
13:57
capacitors been added there
13:59
uh there's a couple of amplifiers prior
14:01
to the HD converter buffers or Andor
14:05
amplifiers taking the CCD
14:08
signal um bit more power spy stuff here
14:11
this is probably all to do with
14:12
generating the various CCD bias voltages
14:14
they uh need quite a lot of exotic
14:18
voltages next board down is just a
14:20
shield and an interconnect
14:24
boardly shielding the power supply stuff
14:26
from some the more sensitive an log
14:28
stuff
14:31
and this is the bottom board which is
14:32
just all power supply
14:34
stuff as you can see tons of capacitors
14:37
or Transformer in
14:42
there and this is the CCD which is a uh
14:47
very impressive thing this is a kf50 100
14:50
originally by true sense then codec
14:53
there ON
14:54
Semiconductor it's 50 megapixels uh the
14:58
only price I could find was this one
15:00
5,000 bucks and the architecture you can
15:03
see there actually two separate halves
15:04
which explains the Dual AZ converters
15:06
and two amplifiers it's literally split
15:08
down the middle and duplicated so it's
15:10
outputting two streams in parallel just
15:13
to keep the bandwidth sensible and it's
15:15
got this amazing um diffraction effects
15:18
it's got micro lenses over the pixel so
15:20
there's there's a bit more Optics going
15:22
on than on a normal
15:25
sensor few more bodges on the CCD board
15:28
including this wire which isn't really
15:29
tacked down very well which is a bit uh
15:32
bit of a mess quite a few bits around
15:34
this board where they've uh tacked
15:36
various bits on which is not super
15:38
impressive looks like CCD drivers on the
15:40
left with those 3 ohm um damping
15:43
resistors on the
15:47
output get a few more little bodges
15:50
around here some of
15:52
the and there's this separator the
15:54
silica gel to keep the moisture down but
15:56
there's this separator that actually
15:58
appears to be cut from piece of
15:59
antistatic
16:04
bag and this sort of thermal block on
16:06
top of this stack of three pel Cola
16:12
modules so as with any Stacks they get
16:16
um larger as they go back towards the
16:18
heat sink because each P's got to not
16:20
only take the heat from the previous but
16:21
also the waste heat which is quite
16:27
significant you see a little temperature
16:29
sensor here that copper block which
16:32
makes contact with the back of the
16:37
CCD and this's the back of the
16:40
pelas this then contacts the heat sink
16:44
on the uh rear there a few thermal pads
16:46
as well for some of the other power
16:47
components on this
16:51
PCB okay I've connected this uh camera
16:54
up I found some drivers on the disc that
16:56
seem to work under Windows 7 couldn't
16:58
get to install under Windows 11 though
17:01
um in the absence of any sort of lens or
17:03
being bothered to the proper amount I've
17:04
just put some f over it and put a little
17:06
pin in there to make a pinhole lens and
17:08
software gives a few options I'm not
17:11
entirely sure what all these are there's
17:12
obviously a clock frequency 22 MHz low
17:15
gain and with PFG no idea what that is
17:19
something something game programmable
17:20
Something game perhaps ver exposure
17:23
types I think focus is just like a
17:25
continuous grab until you tell it to
17:27
stop not entirely sure all these options
17:30
are obviously exposure time uh triggers
17:33
there ex external hardware trigger inut
17:35
you just trigger using a um thing on
17:37
screen so the resolution is 8176 by
17:40
6132 and you can actually bin those
17:42
where you combine multiple pixels to get
17:46
increased gain at the expense of lower
17:48
resolution down this is a 10sec exposure
17:51
obviously of the pin hole it's very uh
17:53
intensitive so we just stand still now
17:56
downloading it there's the uh exposure
17:59
so when it's
18:01
um there's a little status thing down
18:03
here so that tells you the um exposure
18:07
[Applause]
18:09
time it's this is just it
18:15
downloading um it is quite I'm seeing
18:18
quite a lot like smearing I think that I
18:20
don't know whether that's just due to
18:21
pixels overloading or something else I
18:24
mean yeah it's not it's not um out of
18:26
the question that there's something not
18:27
totally right about this camera
18:28
certainly was bodge wise on there um I
18:31
don't I'd imagine a camera like this
18:32
it's got a fairly narrow range of
18:34
intensities that it's happy with I'm not
18:36
going to spend a great deal of time on
18:38
this if you're interested in this camera
18:40
maybe for astronomy or something and
18:42
happy to sort of take the risk of it may
18:44
not be uh perfect I'll um I think I'll
18:47
stick this on eBay along with the
18:48
Illuminator I'll put a link down in the
18:50
description to the listing take your
18:52
chances to grab a bargain so for example
18:54
here we see this vertical streaking so
18:56
I'm not sure how normal that is this is
18:58
on fairly bright scene looking out the
19:02
window if I cut the exposure time down
19:04
on that it's now 1 second
19:07
exposure again most of the image
19:09
disappears again this is looks like it's
19:11
possibly over still overloading here go
19:14
that go down to say say quarter a
19:16
second so again I think there might be
19:19
some Auto gain control going on here um
19:21
this is with the PFG option let's try
19:23
turning that off and see what
19:25
happens so I'm not sure this is actually
19:27
more streaking or which just it's
19:29
cranked up the gain all the dis display
19:31
gray scale to show what um you know the
19:33
range of things that it's captured
19:36
there's one of one of 12 things in the
19:38
software there's um you can see of you
19:40
can't seem to read out the temperature
19:42
of the pelta cooler but you can set the
19:44
temperature and if you said it's a
19:46
different temperature you see the power
19:48
consumption jump up running the cooler
19:50
to get the temperature you requested but
19:52
I can't see anything anywhere that tells
19:54
you whether the cool is at the at the
19:56
temperature other than the power
19:57
consumption going down and there's no
19:59
temperature read out
20:03
here and just some yeah this is just
20:05
sort of very basic software I'm sure
20:07
there's like an API for more
20:09
sophisticated
20:10
applications but so if you know anything
20:12
more about these cameras please um stick
20:14
in the
20:15
comments um incidentally when I was
20:18
editing I didn't notice there was a bent
20:19
pin on the um CCD but I did fix that
20:22
before doing these tests and also
20:24
reactivated the um silica gel desicant
20:26
cuz I noticed it was uh I was getting
20:28
bit of condensation on the window but um
20:31
yeah so a couple of uh interesting but
20:34
maybe not particularly uh useful pieces
20:37
of Kit except for someone that's got a
20:38
very specific use so um I'll stick a
20:42
I'll stick these on eBay put a link in
20:44
the description and say hopefully
20:45
someone could actually make some uh good
20:47
use of these things
Example Output:
**Exploring the Fluidigm Polaris: A Detailed Look at its High-End Optics and Camera System**

* **0:00 High-End Optics:** The system utilizes heavy, high-quality lenses and mirrors for precise imaging, weighing around 4 kilos each.
* **0:49 Narrow Band Filters:** A filter wheel with five narrow band filters (488, 525, 570, 630, and 700 nm) ensures accurate fluorescence detection and rejection of excitation light. 
* **2:01 Customizable Illumination:** The Lumen Dynamics light source offers five individually controllable LED wavelengths (430, 475, 520, 575, 630 nm) with varying power outputs. The 575nm yellow LED is uniquely achieved using a white LED with filtering.
* **3:45 TTL Control:**  The light source is controlled via a simple TTL interface, enabling easy on/off switching for each LED color.
* **12:55 Sophisticated Camera:**  The system includes a 50-megapixel Kodak KAI-50100 CCD camera with a Peltier cooling system for reduced noise.
* **14:54 High-Speed Data Transfer:** The camera features dual analog-to-digital converters to manage the high data throughput of the 50-megapixel sensor, which is effectively two 25-megapixel sensors operating in parallel.
* **18:11 Possible Issues:** The video creator noted some potential issues with the camera, including image smearing. 
* **18:11 Limited Dynamic Range:** The camera's sensor has a limited dynamic range, making it potentially challenging to capture scenes with a wide range of brightness levels.
* **11:45 Low Runtime:** Internal data suggests the system has seen minimal usage, with only 20 minutes of recorded runtime for the green LED.
* **20:38 Availability on eBay:** Both the illuminator and camera are expected to be listed for sale on eBay.
Here is the real transcript. Please summarize it: 
{s.transcript}"""
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
                print("Error")
        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=response.usage_metadata.prompt_token_count, summary_output_tokens=response.usage_metadata.candidates_token_count, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, summary_done=False, summary=((summaries[identifier].summary)+("\nError: resource exhausted")), summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return
    try:
        text=summaries[identifier].summary
        # adapt the markdown to YouTube formatting
        text=text.replace("**:", ":**")
        text=text.replace("**,", ",**")
        text=text.replace("**.", ".**")
        text=text.replace("**", "*")
        # markdown title starting with ## with fat text
        text=re.sub(r"""^##\s*(.*)""", r"""*\1*""", text)
        # find any text that looks like an url and replace the . with -dot-
        text=re.sub(r"""((?:https?://)?(?:www\.)?\S+)\.(com|org|de|us|gov|net|edu|info|io|co\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr)""", r"""\1-dot-\2""", text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=0, timestamps_output_tokens=0, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=f"resource exhausted", timestamps_timestamp_end=datetime.datetime.now().isoformat())
        return
 
serve(host="0.0.0.0", port=5001)