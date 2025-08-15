9| file              | comment                                                                             |
|-------------------+-------------------------------------------------------------------------------------|
| captions.py       | download captions with youtube api (an example), not working for third party videos |
| gapi.py           | grpc based generativeai interface (can't install this on vm)                        |
| gapi2.py          | REST based gemini transcript summarization                                          |
| get_transcript.py | another failed attempt to get video transcripts                                     |
| run.py            | an attempt to use gemini via selenium (very slow and brittle for large transcripts) |
| selen_comments.py | use selenium to download all my youtube comments                                    |
| gen01             | try to generate fasthtml interface with storage in files                            |
| gen02             | try to generate fasthtml interface with storage in sqlite based on gapi2.py and 144 |
| gen03             | experiment with google-genai  and yt-dlp                                            |
| gen04             | split up 02 for testing   (this is what i run most of the time)                     |


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


# prepare gemini embeddings for t-SNE or UMAP

https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/clustering_with_embeddings.ipynb
5
https://news.ycombinator.com/item?id=38976773

This Hacker News discussion revolves around K-Means clustering, sparked by a post sharing a generalized, scalable implementation using Spark. Commenters explore various facets of K-Means and related techniques, particularly in the context of modern AI applications involving high-dimensional embeddings. Key themes include practical pipelines combining text embedding (e.g., SentenceTransformers), dimensionality reduction (UMAP), clustering (K-Means, DBSCAN, HDBSCAN), and automated labeling using LLMs like ChatGPT. Significant discussion addresses the challenge of selecting the optimal number of clusters (k), with users suggesting heuristics like cluster stability, Silhouette index, and Calinski-Harabasz index, or advocating for alternatives like DBSCAN/HDBSCAN that don't require pre-specifying k. The conversation also covers diverse applications of K-Means (text analysis, image processing, GIS, e-commerce, bioinformatics, noise reduction), performance considerations for large datasets, implementation details (libraries, specific models), the algorithm's theoretical underpinnings (relation to EM/GMM), and its perceived simplicity versus limitations compared to other methods.



# Fail2Ban

- too many clients try to find PHP vulnerabilities

```

sudo apt update && sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```


# Nginx http/3

i am trying to compile nginx with http/3 support on ubuntu 24.04

```
sudo apt install gcc g++ make libpcre3-dev zlib1g-dev ninja-build cmake
git clone https://github.com/google/boringssl
cd boringssl
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

git clone https://github.com/nginx/nginx


cd nginx
auto/configure \
--with-http_ssl_module \
--with-http_v3_module \
--with-cc-opt="-I../boringssl/include" \
--with-ld-opt="-L../boringssl/build -lstdc++" 

make -j2 
make install

```

the configure step fails with:

```
auto/configure --with-http_ssl_module --with-http_v3_module --with-cc-opt="-I../boringssl/include" --with-ld-opt="-L../boringssl/build -lstdc++"
checking for OS
 + Linux 6.8.0-58-generic x86_64
checking for C compiler ... found
 + using GNU C compiler
 + gcc version: 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
checking for gcc -pipe switch ... found
checking for --with-ld-opt="-L../boringssl/build -lstdc++" ... found
checking for -Wl,-E switch ... found
...
checking for PCRE2 library ... not found
checking for PCRE library ... found
checking for PCRE JIT support ... found
checking for OpenSSL library ... not found
checking for OpenSSL library in /usr/local/ ... not found
checking for OpenSSL library in /usr/pkg/ ... not found
checking for OpenSSL library in /opt/local/ ... not found
checking for OpenSSL library in /opt/homebrew/ ... not found

auto/configure: error: SSL modules require the OpenSSL library.
You can either do not enable the modules, or install the OpenSSL library
into the system, or build the OpenSSL library statically from the source
with nginx by using --with-openssl=<path> option.
```


here is some documentation from the nginx website:

Support for QUIC and HTTP/3 protocols is available since 1.25.0. Also, since 1.25.0, the QUIC and HTTP/3 support is available in Linux binary packages.


The QUIC and HTTP/3 support is experimental, caveat emptor applies.

Building from sources
The build is configured using the configure command. Please refer to Building nginx from Sources for details.

When configuring nginx, it is possible to enable QUIC and HTTP/3 using the --with-http_v3_module configuration parameter.

An SSL library that provides QUIC support is recommended to build nginx, such as BoringSSL, LibreSSL, or QuicTLS. Otherwise, the OpenSSL compatibility layer will be used that does not support early data.

Use the following command to configure nginx with BoringSSL:

./configure
    --with-debug
    --with-http_v3_module
    --with-cc-opt="-I../boringssl/include"
    --with-ld-opt="-L../boringssl/build -lstdc++"

Alternatively, nginx can be configured with QuicTLS:

./configure
    --with-debug
    --with-http_v3_module
    --with-cc-opt="-I../quictls/build/include"
    --with-ld-opt="-L../quictls/build/lib"

Alternatively, nginx can be configured with a modern version of LibreSSL:

./configure
    --with-debug
    --with-http_v3_module
    --with-cc-opt="-I../libressl/build/include"
    --with-ld-opt="-L../libressl/build/lib"

After configuration, nginx is compiled and installed using make.

Configuration
The listen directive in ngx_http_core_module module got a new parameter quic which enables HTTP/3 over QUIC on the specified port.

Along with the quic parameter it is also possible to specify the reuseport parameter to make it work properly with multiple workers.

For the list of directives, see ngx_http_v3_module.

To enable address validation:

quic_retry on;
To enable 0-RTT:

ssl_early_data on;
To enable GSO (Generic Segmentation Offloading):

quic_gso on;
To set host key for various tokens:

quic_host_key <filename>;

QUIC requires TLSv1.3 protocol version which is enabled by default in the ssl_protocols directive.

By default, GSO Linux-specific optimization is disabled. Enable it in case a corresponding network interface is configured to support GSO.

Example Configuration

http {
    log_format quic '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent" "$http3"';

    access_log logs/access.log quic;

    server {
        # for better compatibility it's recommended
        # to use the same port for quic and https
        listen 8443 quic reuseport;
        listen 8443 ssl;

        ssl_certificate     certs/example.com.crt;
        ssl_certificate_key certs/example.com.key;

        location / {
            # required for browsers to direct them to quic port
            add_header Alt-Svc 'h3=":8443"; ma=86400';
        }
    }
}

Troubleshooting
Tips that may help to identify problems:

Ensure nginx is built with the proper SSL library.
Ensure nginx is using the proper SSL library in runtime (the nginx -V shows what it is currently used).
Ensure a client is actually sending requests over QUIC. It is recommended to start with a simple console client such as ngtcp2 to ensure the server is configured properly before trying with real browsers that may be quite picky with certificates.
Build nginx with debug support and check the debug log. It should contain all details about the connection and why it failed. All related messages contain the “quic” prefix and can be easily filtered out.
For a deeper investigation, additional debugging can be enabled using the following macros: NGX_QUIC_DEBUG_PACKETS, NGX_QUIC_DEBUG_FRAMES, NGX_QUIC_DEBUG_ALLOC, NGX_QUIC_DEBUG_CRYPTO.

./configure
    --with-http_v3_module
    --with-debug
    --with-cc-opt="-DNGX_QUIC_DEBUG_PACKETS -DNGX_QUIC_DEBUG_CRYPTO"


# Live Chat

## Getting

```
yt-dlp --cookies-from-browser chrome --skip-download --write-subs --sub-langs live_chat
```


## Parsing

```
$ jq -r '
  .replayChatItemAction.actions[].addChatItemAction.item.liveChatTextMessageRenderer
  | select(. != null)
  | [.timestampText.simpleText, .authorName.simpleText, .message.runs[0].text]
  | @tsv
' < GPT5_Code-\[xiJwgwoiXpc\].live_chat.json 
0:13    Abdeljalil ELMAJJODI    Hello hello
0:23    NLP Prompter    it is sometimes funny to think... that we need so many different AI to work on something
0:37    Sorry dave, I am afraid i can't do that.        yoo
0:48    NLP Prompter    i can't imagine to add text into image without gpt or qwen image now
1:27    Sorry dave, I am afraid i can't do that.        I was waiting for your video
2:44    dm204375        cursor now has a cli
2:50    Josh Phillips   oh sweet! I've been meaning to look into opencode. It looked interesting
3:48    NLP Prompter    ELON MUSK 

```