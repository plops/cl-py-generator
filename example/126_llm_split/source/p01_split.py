#!/usr/bin/env python3
# python -m venv ~/gpt_env; . ~/gpt_env/bin/activate; python -m pip install tiktoken
import time
import argparse
start_time=time.time()
debug=True
_code_git_version="1bea3f457ab060a8afa7e323032d0ee48da2522c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/126_llm_split/source/"
_code_generation_time="13:20:10 of Saturday, 2024-04-20 (GMT+1)"
def split_document(input_file, chunk_size, prompt):
    """given an input file split it into several files with at most chunk_size words each. prepend with prompt. replace newlines with space."""
    with open(input_file, "r") as f:
        text=f.read()
    words=text.split()
    chunks=[]
    current_chunk=[]
    word_count=0
    for word in words:
        current_chunk.append(word)
        word_count += 1
        if ( ((chunk_size)<=(word_count)) ):
            chunks.append(" ".join(current_chunk))
            current_chunk=[]
            word_count=0
    if ( current_chunk ):
        chunks.append(" ".join(current_chunk))
    for i, chunk in enumerate(chunks):
        output_file="{}.{}".format(input_file.split(".")[0], str(((i)+(1))).zfill(2))
        with open(output_file, "w") as f:
            f.write(((prompt)+("\n```\n")))
            f.write(chunk)
            f.write("\n```")
if ( ((__name__)==("__main__")) ):
    parser=argparse.ArgumentParser(description="Split a document into chunks. I use this to make summaries with chatgpt4")
    parser.add_argument("input_file", type=str, help="The input text file to split.")
    parser.add_argument("-c", "--chunk_size", help="Approximate number of words per chunk", default=500, type=int, action=None)
    parser.add_argument("-p", "--prompt", help="The prompt to be prepended to the output file(s).", default="Summarize the following video transcript as a bullet list.", type=str, action=None)
    args=parser.parse_args()
    split_document(args.input_file, args.chunk_size, args.prompt)