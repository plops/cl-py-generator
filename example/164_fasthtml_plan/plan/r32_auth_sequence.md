Of course. Here is the text section for the "User Authentication Flow," including a description and the corresponding Mermaid sequence diagram.

***

### 2. User Authentication Flow (Sequence Diagram)

A sequence diagram is the ideal tool to visualize the time-ordered interactions between different participants in a system. This diagram illustrates the standard OAuth 2.0 Authorization Code Grant flow, which is used to securely log a user into RocketRecap using their existing Google account.

The process involves a three-way communication between the user's browser, the RocketRecap server, and the Google OAuth provider. The key security aspect is that the user's Google credentials are never exposed to the RocketRecap application. Instead, a temporary authorization code is used to securely fetch the user's profile information after they have given consent.

```mermaid
sequenceDiagram
    actor User
    participant RocketRecap Server
    participant Google (OAuth Provider)

    User->>RocketRecap Server: Clicks 'Sign in with Google' on /login
    RocketRecap Server-->>User: HTTP 302 Redirect to Google's consent screen

    User->>Google (OAuth Provider): Logs in and grants permissions
    Google (OAuth Provider)-->>User: HTTP 302 Redirect to /auth/google/callback?code=...

    User->>RocketRecap Server: GET /auth/google/callback?code=...
    activate RocketRecap Server

    Note over RocketRecap Server, Google (OAuth Provider): Backend Communication
    RocketRecap Server->>Google (OAuth Provider): Exchange authorization code for access token
    Google (OAuth Provider)-->>RocketRecap Server: Returns access token

    RocketRecap Server->>Google (OAuth Provider): Request user profile with access token
    Google (OAuth Provider)-->>RocketRecap Server: Returns user profile (email, id)

    Note over RocketRecap Server: AuthService Logic
    RocketRecap Server->>RocketRecap Server: Get or Create User record in PostgreSQL DB

    RocketRecap Server-->>User: Set secure session cookie & HTTP 303 Redirect to /
    deactivate RocketRecap Server

    User->>RocketRecap Server: GET / (request with session cookie)
    RocketRecap Server-->>User: Renders personalized dashboard

```
