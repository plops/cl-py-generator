
# Description

This project allows you to programmatically interact with Google Bard,
a large language model (LLM) developed by Google AI. It utilizes an
unofficial Python library called bardapi to simplify communication
with Bard.


# Setup Python Dependencies

1. Create a virtual environment:

```bash
python -m venv ~/bardapi-venv
```

2. Activate the virtual environment:

```bash
source ~/bardapi-venv/bin/activate
```

3. Install required dependencies:

```bash
pip install bardapi toml
```

4. Deactivate the virtual environment when you're done:

```bash
deactivate
```


# Handling Cookies

1. *Obtain Cookies:* Watch this video to learn how to obtain the
   necessary cookies from your browser using the developer tools:
   [https://www.youtube.com/watch?v=1a6-QrmYf1M](https://www.youtube.com/watch?v=1a6-QrmYf1M)

2. *Create Cookie File:* Create a file named `env.toml` in the same
   directory as the Python script. Enter the obtained cookies into the
   file in the following format:

```toml
[cookies]
__Secure-1PSID = "dQi..."
__Secure-1PSIDCC = "ACB-..."
__Secure-1PAPISID = "Xx..."
```

*Note:*

For security reasons, avoid committing the `env.toml` file to GitHub.


# Comparison Bard and GPT4

## Summarize as bullet list 

I give a video transcript and ask for a summary

### Bard:
```
The video explains how to convert a video into densepose video. The steps are:
1. Download the code from the GitHub repository.
2. Install the requirements.
3. Download the models.
4. Run the command `python convert.py`.
5. Provide the input video path and output video path.
6. The conversion process will start and take a few minutes.
7. The output video will be created in the same directory.

```

### GPT4:
```
The video is about converting an input video into a dense pose video using Detectron 2 and DensPose.
The presenter provides a GitHub repository link for downloading the code and instructions for installation.
The process involves downloading and installing requirements, downloading the models, and running a Python script to convert the input video.
The presenter encourages contributions to the repository and mentions potential inefficiencies in the code.
Viewers are advised to seek help in the comments section if they encounter any issues during the process.
```


### Bard with one example:

```
- Download the One Click DensePose code from the GitHub repository link.
- Install the requirements by running the provided command.
- Download the models by running the provided shell script command.
- Run the conversion script by providing the input video path and output path.
- The conversion process will take a few minutes depending on your laptop's performance.
- If you are interested in stopping the conversion process, you can hit control+C.
- The output video will be saved in the same directory with the name "out.mp4".

```

# Python interface for OpenAI API

https://github.com/lablab-ai/openai-chat-endpoint-boilerplate

- even with the free openai account  i was able to create an api key
