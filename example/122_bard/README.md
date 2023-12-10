
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

