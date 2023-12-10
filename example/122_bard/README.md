
# Description

This program talks to google bard using an unofficial google bard api.


# Setup python dependencies

create and enter the python environment
```
python -m venv ~/bardapi-venv
source ~/bardapi-venv/bin/activate
pip install bardapi toml
```

leave the python environment like this

```
deactivate
```

# How to handle cookies

This video shows how to obtain the cookies
https://www.youtube.com/watch?v=1a6-QrmYf1M using the browser
development tools.


Create file named env.toml in the same directory as the python script:
```
[cookies]
__Secure-1PSID = "dQi..."
__Secure-1PSIDCC = "ACB-..."
__Secure-1PAPISID = "Xx..."
```

Try not to publish this file on github.
