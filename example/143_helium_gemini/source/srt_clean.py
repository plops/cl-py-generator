import re

def clean_srt(srt_content):
    lines = srt_content.splitlines()
    cleaned_lines = []
    current_text = ""

    i = 0
    while i < len(lines):
        if re.match(r"\d+", lines[i]):  # Check for index line (e.g., "1", "2")
            i += 1  # Skip the index line
            timestamp_line = lines[i]
            match = re.match(r"(\d\d:\d\d:\d\d,\d+)", timestamp_line)  # Extract start time
            timestamp = match.group(1).split(",")[0] if match else "" 

            i += 2  # Skip the timestamps and the usually present immediately succeeding blank line

            text_lines = []
            while i < len(lines) and not re.match(r"\d+", lines[i]) and lines[i].strip() != "":
                text_lines.append(lines[i])
                i += 1

            text = "".join(text_lines).strip()


            if text != current_text and text != "":  # Check both for repetition AND empty string!!! IMPORTANT
                cleaned_lines.append(f"{timestamp} {text}")
                current_text = text  

            while i < len(lines) and lines[i].strip() == "":  #handle optional blank lines after a section
                i += 1

        else:  # Increment otherwise as well to catch starting indices missed due to above if being False, for edge cases etc
            i += 1
    return "\n".join(cleaned_lines)

srt_content = """
1
00:00:00,160 --> 00:00:02,470

this week in microbiology is brought to

2
00:00:02,470 --> 00:00:02,480
this week in microbiology is brought to

3
00:00:02,480 --> 00:00:05,030
this week in microbiology is brought to
you by the American Society for

4
00:00:05,030 --> 00:00:05,040
you by the American Society for

5
00:00:05,040 --> 00:00:08,030
you by the American Society for
microbiology at amm.org

6
00:00:08,030 --> 00:00:08,040
microbiology at amm.org

7
00:00:08,040 --> 00:00:09,630
microbiology at amm.org
SL

8
00:00:09,630 --> 00:00:09,640
SL

9
00:00:09,640 --> 00:00:19,470
SL
[Music]

10
00:00:19,470 --> 00:00:19,480
[Music]

11
00:00:19,480 --> 00:00:23,390
[Music]
twim this is twim this weekend

12
00:00:23,390 --> 00:00:23,400
twim this is twim this weekend

13
00:00:23,400 --> 00:00:25,509
twim this is twim this weekend
microbiology where today we are

14
00:00:25,509 --> 00:00:25,519
microbiology where today we are

15
00:00:25,519 --> 00:00:28,269
microbiology where today we are
recording a very special episode from

16
00:00:28,269 --> 00:00:28,279
recording a very special episode from
"""



cleaned_srt = clean_srt(srt_content)
print(cleaned_srt)


# Here's how the improved version works:


# 1. **Iterates line by line:** Instead of fixed blocks, the code now iterates through lines, more flexibly handling varying text lengths.

# 2. **Identifies start of a block**: It detects the beginning of a new block by the numerical index (e.g., "1", "2", etc.).

# 3. **Extracts timestamp:** Captures timestamp  as before.

# 4. **Gathers text lines:** Accumulates all lines of text belonging to a block *until* it encounters the next numerical index or a completely empty line which separates next block , effectively dealing with multi-line text correctly. It also works with single-lined text blocks as in initial example you provided .

# 5. **Filters repetitions:** The core de-duplication logic remains the same using the  comparison against `current_text`.

# 6. **Handles Blank lines separating blocks** Blank lines in srt signal a switch between one caption section with its timecode and next section. Text usually end just before a blank line , with timestamps showing above and after blank lines at section start .Blank lines may or may not show in input .We need to cater to varying cases to correctly switch between processing consecutive srt chunks including cases such as two consecutive blank lines before new timestamps in the input,  for instance. Now code only adds to output what it collected only after the entire block is constructed (ie., when the complete multi line or single line content pertaining to some timestamp is all obtained by code). Output addition doesn't occur anymore  repeatedly before blocks finish building (since in that original case , that leads to repeat lines corresponding to unfinished sections of timestamps) 
