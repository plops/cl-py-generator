Okay, let's enable QUIC (HTTP/3) for your `fasthtml` application using Nginx.

**1. Prerequisite: Nginx Version**

The biggest hurdle is your current Nginx version:

```
nginx version: nginx/1.24.0 (Ubuntu)
```

According to the Nginx documentation you provided, QUIC and HTTP/3 support was introduced in **version 1.25.0**.

**Therefore, you MUST upgrade Nginx to version 1.25.0 or later.**

**How to Upgrade Nginx (Choose one):**

*   **Recommended (Easier): Use a PPA or Backports (Ubuntu/Debian)**
    *   The most common way to get newer Nginx versions on Ubuntu is often through Ondřej Surý's PPA:
        ```bash
        sudo add-apt-repository ppa:ondrej/nginx-mainline # Or ppa:ondrej/nginx for stable
        sudo apt update
        sudo apt install nginx
        nginx -v # Verify the version is >= 1.25.0
        ```
    *   Alternatively, check if your Ubuntu version has a backports repository that includes a sufficiently new Nginx.
*   **Build from Source (More Complex):**
    *   If using a PPA isn't possible or desired, you'll need to compile Nginx from source, ensuring you include the `--with-http_v3_module` flag and link against a compatible SSL library (like BoringSSL, LibreSSL, or QuicTLS) as shown in the documentation you provided. This is significantly more involved.

**Assuming you have successfully upgraded Nginx to >= 1.25.0:**

**2. Update Nginx Configuration (`nginx.conf`)**

You need to modify the `server` block that handles HTTPS (port 443) to also listen for QUIC connections on the same port and advertise its availability.

```nginx
# the fasthtml library hosts the website using the uvicorn library via
# http on 127.0.0.1:5001. Nginx handles TLS and QUIC.

events {}

http {
    # Add $http3 variable to log format to see which protocol is used
    log_format quic '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent" "$http3"'; # Added $http3

    # Use the new log format
    access_log /var/log/nginx/access.log quic; # Or your preferred log path
    error_log /var/log/nginx/error.log warn;   # Or your preferred log path/level

    server {
        listen 80;
        listen [::]:80;
        server_name rocketrecap.com;

        # Optional: Let Let's Encrypt/Certbot handle HTTP challenges
        location /.well-known/acme-challenge/ {
            root /var/www/html; # Or your certbot webroot path
        }

        location / {
           return 301 https://$server_name$request_uri;
        }
    }

    server {
        # Listen on TCP port 443 for HTTPS (HTTP/1.1, HTTP/2)
        listen 443 ssl http2;       # Added http2 for potentially better fallback
        listen [::]:443 ssl http2;  # Added http2 for potentially better fallback

        # Listen on UDP port 443 for QUIC (HTTP/3)
        # reuseport is recommended with QUIC and multiple workers
        listen 443 quic reuseport;
        listen [::]:443 quic reuseport;

        server_name rocketrecap.com;

        # --- SSL Certificates ---
        ssl_certificate      /etc/letsencrypt/live/rocketrecap.com/fullchain.pem;
        ssl_certificate_key  /etc/letsencrypt/live/rocketrecap.com/privkey.pem;
        # Optional: Add recommended SSL parameters
        ssl_protocols TLSv1.2 TLSv1.3; # QUIC requires TLSv1.3
        ssl_prefer_server_ciphers off;
        ssl_session_timeout 1d;
        ssl_session_cache shared:SSL:10m;  # Or specify your cache size
        ssl_session_tickets off; # Consider security implications if enabling

        # --- QUIC/HTTP3 Specific Settings ---
        # Enable address validation (recommended)
        quic_retry on;

        # Optional: Enable 0-RTT. Requires client support and understanding security implications.
        # ssl_early_data on;

        # --- Gzip Compression Settings (Keep as is) ---
        gzip on;
        gzip_vary on;
        gzip_proxied any;
        gzip_comp_level 6;
        gzip_buffers 16 8k;
        gzip_http_version 1.1;
        gzip_min_length 1000;
        gzip_types
            text/plain
            text/css
            application/json
            application/javascript
            application/x-javascript
            text/xml
            application/xml
            application/xml+rss
            image/svg+xml;

        location / {
            # --- Advertise QUIC/HTTP3 availability ---
            # Tells browsers HTTP/3 is available on UDP port 443.
            # ma=86400 means cache this info for 1 day (adjust as needed)
            # 'always' ensures the header is added even on error responses
            add_header Alt-Svc 'h3=":443"; ma=86400' always;

            # --- Proxy Settings (Keep as is) ---
            proxy_pass http://127.0.0.1:5001;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme; # Good practice
            proxy_http_version 1.1; # Ensure HTTP/1.1 to backend
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

**Key Changes:**

1.  **Nginx Version:** Upgraded to >= 1.25.0.
2.  **`log_format`:** Added `$http3` variable to see `h3` in logs for QUIC connections.
3.  **`listen` directives:**
    *   Added `http2` to the `listen ... ssl` directives for better performance if QUIC fails.
    *   Added `listen 443 quic reuseport;` and `listen [::]:443 quic reuseport;` to listen on UDP port 443.
4.  **`ssl_protocols`:** Ensured `TLSv1.3` is included, as it's required by QUIC.
5.  **`quic_retry on;`:** Added to enable address validation, improving robustness against spoofing.
6.  **`add_header Alt-Svc ...`:** Added *inside* the `location /` block. This header is crucial; it's sent over the regular HTTPS connection to tell the browser that HTTP/3 is available on UDP port 443.
7.  **`proxy_set_header X-Forwarded-Proto $scheme;`**: Added as good practice so the backend knows the original request scheme (http/https).
8.  **`proxy_http_version 1.1;` etc.:** Ensured backend connection uses standard HTTP/1.1, which `uvicorn` understands.

**3. Firewall Configuration**

QUIC runs over UDP. Your firewall must allow incoming traffic on **UDP port 443**. TCP port 443 should already be open for HTTPS.

*   **If using `ufw`:**
    ```bash
    sudo ufw allow 443/udp comment 'Allow QUIC/HTTP3'
    sudo ufw reload
    ```
*   **If using `iptables`:** The command will be different, depending on your specific ruleset, but you'll need to add a rule similar to:
    ```bash
    sudo iptables -I INPUT -p udp --dport 443 -j ACCEPT
    # Make sure to save the rules if necessary (e.g., using iptables-persistent)
    ```

**4. Test Configuration and Reload Nginx**

*   Test your Nginx configuration for syntax errors:
    ```bash
    sudo nginx -t
    ```
*   If the test is successful, reload Nginx to apply the changes:
    ```bash
    sudo systemctl reload nginx
    ```

**5. Verify QUIC/HTTP/3 is Working**

*   **Browser Developer Tools:** Open your website (e.g., `https://rocketrecap.com`) in a modern browser (Chrome, Firefox, Edge). Open the Developer Tools (F12), go to the "Network" tab, reload the page (Ctrl+R or Cmd+R). Check the "Protocol" column. You should see "h3" or "HTTP/3" for the main requests. You might need to right-click the column headers and enable the "Protocol" column if it's not visible.
*   **Online Checkers:** Use websites like `http3check.net` or `geekflare.com/tools/http3-test` and enter your domain (`rocketrecap.com`). They will test if your server supports QUIC/HTTP/3.
*   **Nginx Logs:** Check your Nginx access log (e.g., `/var/log/nginx/access.log`). Thanks to the updated `log_format`, you should see `h3` at the end of log lines for requests served over HTTP/3.

**No Changes Needed in Python Code**

Your `fasthtml` application (`p04_host.py`) running on `127.0.0.1:5001` continues to speak plain HTTP/1.1. Nginx handles all the complexity of TLS termination (for HTTPS) and QUIC termination (for HTTP/3), forwarding a standard HTTP request to your Python backend.
