Of course. Here is the text section for the "Asynchronous Summarization Process," which details the most complex and dynamic workflow in the application.

***

### 3. Asynchronous Summarization Process (Sequence Diagram)

This diagram visualizes the end-to-end, asynchronous workflow for generating a video summary. A key architectural decision for RocketRecap v2 is to provide a non-blocking, real-time user experience. This is achieved by immediately responding to the user's request while initiating a long-running background task.

The client-side UI, powered by HTMX, establishes a Server-Sent Events (SSE) connection to receive live updates. The background workflow communicates its progress (e.g., "Downloading," "Generating") and streams the AI-generated summary back to the user as it's being created. This is all mediated by an in-memory Pub/Sub service (`SSEService`) to decouple the background task from the web request and to avoid overwhelming the database with constant small writes.

```mermaid
sequenceDiagram
    actor User (Browser)
    participant Server
    participant SummarizationWorkflow (Background Task)
    participant SSEService (In-Memory Pub/Sub)
    participant PostgreSQL DB
    participant TranscriptProvider
    participant GenAIAdapter

    User->>Server: POST /summarize with URL
    activate Server
    Server->>PostgreSQL DB: INSERT new job (status: PENDING)
    PostgreSQL DB-->>Server: Returns new job_id
    Server->>SummarizationWorkflow: start_workflow(job_id)
    Server-->>User: Returns initial HTML with <div sse-connect="/sse/{job_id}">
    deactivate Server

    User->>Server: GET /sse/{job_id} (establishes SSE connection)
    Note over Server, SSEService: Server subscribes to SSEService for job_id

    activate SummarizationWorkflow
    SummarizationWorkflow->>PostgreSQL DB: UPDATE job status to DOWNLOADING
    SummarizationWorkflow->>SSEService: publish(job_id, {status: 'DOWNLOADING'})
    SSEService-->>User: Pushes SSE status update event

    SummarizationWorkflow->>TranscriptProvider: get_from_youtube(url)
    activate TranscriptProvider
    TranscriptProvider-->>SummarizationWorkflow: Returns transcript text
    deactivate TranscriptProvider

    SummarizationWorkflow->>PostgreSQL DB: UPDATE job status to GENERATING
    SummarizationWorkflow->>SSEService: publish(job_id, {status: 'GENERATING'})
    SSEService-->>User: Pushes SSE status update event

    SummarizationWorkflow->>GenAIAdapter: generate_summary_stream(transcript)
    activate GenAIAdapter
    loop For each chunk in AI stream
        GenAIAdapter-->>SummarizationWorkflow: Yields content chunk
        SummarizationWorkflow->>SSEService: publish(job_id, {content: 'New text...'})
        SSEService-->>User: Pushes SSE content event for HTMX to append
    end
    deactivate GenAIAdapter
    
    Note over SummarizationWorkflow, PostgreSQL DB: Perform final batched DB write
    SummarizationWorkflow->>PostgreSQL DB: UPDATE summary_markdown with full text
    SummarizationWorkflow->>PostgreSQL DB: UPDATE job status to COMPLETED, save final costs
    
    SummarizationWorkflow->>SSEService: publish(job_id, {status: 'COMPLETED'})
    SSEService-->>User: Pushes final SSE completion event
    
    SummarizationWorkflow->>SSEService: close_channel(job_id)
    deactivate SummarizationWorkflow
```
