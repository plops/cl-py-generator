| file              | comment                                                                              |
|                   |                                                                                      |
| captions.py       | download captions with youtube api (an example), not working for third party videos  |
| gapi.py           | grpc based generativeai interface (can't install this on vm)                         |
| gapi2.py          | REST based gemini transcript summarization                                           |
| get_transcript.py | another failed attempt to get video transcripts                                      |
| run.py            | an attempt to use gemini via selenium (very slow and brittle for large transcripts)  |
| selen_comments.py | use selenium to download all my youtube comments                                     |
| gen01             | try to generate fasthtml interface with storage in files                             |
| gen02             | try to generate fasthtml interface with storage in sqlite  based on gapi2.py and 144 |
|                   |                                                                                      |


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
