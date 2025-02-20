| file              | comment                                                                                                                   |
|-------------------+---------------------------------------------------------------------------------------------------------------------------|
| captions.py       | download captions with youtube api (an example), not working for third party videos                                       |
| gapi.py           | grpc based generativeai interface (can't install this on vm)                                                              |
| gapi2.py          | REST based gemini transcript summarization                                                                                |
| get_transcript.py | another failed attempt to get video transcripts                                                                           |
| run.py            | an attempt to use gemini via selenium (very slow and brittle for large transcripts)                                       |
| selen_comments.py | use selenium to download all my youtube comments                                                                          |
| gen01             | try to generate fasthtml interface with storage in files                                                                  |
| gen02             | try to generate fasthtml interface with storage in sqlite based on gapi2.py and 144 (this is what i run most of the time) |
| gen03             | experiment with google-genai  and yt-dlp                                                                                  |
| gen04             | split up 02 for testing                                                                                                   |


# Create a proxy 

- the free version of gemini api works better if accessed from us

- solution:
  - rent a server in the us
  - create socks proxy: `ssh -D 5555 server-in-us`
  - run tinyproxy (this is needed because grpc can't go through socks
    ('socks5h' scheme not supported in proxy URI)
	- documentation: https://tinyproxy.github.io/


```
# tinyproxy.conf

Port 8888
Listen 127.0.0.1
Timeout 600

# Use the SOCKS proxy running on localhost:5555
Upstream socks5 127.0.0.1:5555

# Only allow connections from localhost
Allow 127.0.0.1
```

   - run `tinyproxy -d -c tinyproxy.conf`

 - `export http_proxy="http://localhost:8888"`

```
cd source02
tinyproxy -d -c tinyproxy.conf &
export http_proxy="http://localhost:8888"
python -i ./p01_host.py 
```

here is the tinyproxy output:

```
NOTICE    Aug 25 12:31:18.208 [26268]: Initializing tinyproxy ...
NOTICE    Aug 25 12:31:18.208 [26268]: Reloading config file (tinyproxy.conf)
INFO      Aug 25 12:31:18.208 [26268]: Added address [127.0.0.1] to listen addresses.
INFO      Aug 25 12:31:18.208 [26268]: Added upstream socks5 127.0.0.1:5555 for [default]
NOTICE    Aug 25 12:31:18.208 [26268]: Reloading config file finished
INFO      Aug 25 12:31:18.208 [26268]: listen_sock called with addr = '127.0.0.1'
INFO      Aug 25 12:31:18.208 [26268]: trying to listen on host[127.0.0.1], family[2], socktype[1], proto[6]
INFO      Aug 25 12:31:18.208 [26268]: listening on fd [3]
INFO      Aug 25 12:31:18.208 [26268]: Not running as root, so not changing UID/GID.
INFO      Aug 25 12:31:18.208 [26268]: Setting the various signals.
INFO      Aug 25 12:31:18.208 [26268]: Starting main loop. Accepting connections.
CONNECT   Aug 25 12:34:09.011 [26268]: Connect (file descriptor 4): 127.0.0.1
CONNECT   Aug 25 12:34:09.011 [26268]: Request (file descriptor 4): CONNECT generativelanguage.googleapis.com:443 HTTP/1.1
INFO      Aug 25 12:34:09.011 [26268]: Found upstream proxy socks5 127.0.0.1:5555 for generativelanguage.googleapis.com
INFO      Aug 25 12:34:09.011 [26268]: opensock: opening connection to 127.0.0.1:5555
INFO      Aug 25 12:34:09.012 [26268]: opensock: getaddrinfo returned for 127.0.0.1:5555
CONNECT   Aug 25 12:34:09.012 [26268]: Established connection to socks5 proxy "127.0.0.1" using file descriptor 5.
INFO      Aug 25 12:34:09.387 [26268]: Not sending client headers to remote machine

```


- explore the sqlite table
```
sqlite3 data/summaries.db
.tables
.schema items
.headers on
SELECT * FROM items ORDER BY identifier DESC LIMIT 1;

```


# yt-dlp usage

## comments
```

# dump complete JSON with comments
yt-dlp --write-comments --dump-single-json -o "$file" "$url"
yt-dlp --write-comments --dump-single-json "$url" > "$file"x
```



# get cookies

```
yt-dlp --cookies-from-browser chrome --skip-download --cookies ff2_cookies.txt; head -n3 ff2_cookies.txt > yt_cookies.txt ; grep ^.youtube.com ff2_cookies.txt >> yt_cookies.txt ; 
scp -Cr yt_cookies.txt ...
```

# instrumentation

by analizing the response time for around 700 summaries (not all of them gave valid results) i found that most of the responses come with 0.2 to 2 seconds per 1000 input tokens.

it might be interesting to store the time when each response packet arrives


# Links

if i copy the link on the ios youtube app it looks like that:

https://www.youtube.com/live/ZOw5LT2vs9E?feature=shared

on the desktop this is the link (what i currently support)
https://www.youtube.com/watch?v=ZOw5LT2vs9E

another video has this link for the pattern:

https://youtu.be/tW6Y1WYwozk?feature=shared


oauth with google and fasthtml

https://www.danliden.com/notebooks/web_dev/fasthtml/5_google_oauth_1.html
