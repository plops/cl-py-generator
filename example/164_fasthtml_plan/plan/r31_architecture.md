Of course. Here is the text section for the "High-Level Architecture" diagram, including a description and the Mermaid diagram code.

***

### 1. High-Level Architecture (Component Diagram)

This diagram provides a high-level overview of the major components of the RocketRecap v2 application, illustrating the layered architecture described in the design document. It shows the clear separation of concerns between handling user interactions (Presentation Layer), orchestrating business logic (Service Layer), and communicating with external systems (Infrastructure Layer). This structure enhances maintainability, testability, and scalability.

The flow begins with the user interacting with the Presentation Layer, which then delegates actions to the Service Layer. The Service Layer orchestrates complex workflows, such as summarization, by utilizing adapters in the Infrastructure Layer to communicate with external dependencies like the database and AI models.

```mermaid

graph TD
    subgraph External Systems
        youtube[<i class='fa fa-youtube'></i> YouTube]
        genai_api[<i class='fa fa-robot'></i> Google GenAI API]
        oauth_provider[<i class='fa fa-google'></i> OAuth Provider]
    end

    subgraph RocketRecap v2 Application
        subgraph Presentation Layer [Presentation Layer - FastHTML]
            direction TB
            routes[Routes & Middleware] --> ui_components[UI Components]
        end

        subgraph Service Layer [Service Layer - Business Logic]
            direction TB
            summarization_workflow[SummarizationWorkflow]
            auth_service[AuthService]
            sse_service[SSEService]
            localization_service[LocalizationService]
        end

        subgraph Infrastructure Layer [Infrastructure Layer - Adapters]
            direction TB
            db_adapter["Database Adapter <br/>(Repositories)"]
            genai_adapter[GenAIAdapter]
            transcript_provider[TranscriptProvider]
        end

        subgraph Domain
            models["Data Models <br/>(Dataclasses)"]
        end
    end

    %% --- Interactions ---
    routes --> auth_service
    routes --> summarization_workflow
    routes --> sse_service

    summarization_workflow --> transcript_provider
    summarization_workflow --> genai_adapter
    summarization_workflow --> db_adapter
    summarization_workflow --> sse_service
    summarization_workflow --> localization_service

    auth_service --> db_adapter
    auth_service --> oauth_provider

    transcript_provider --> youtube
    genai_adapter --> genai_api
    db_adapter --> postgresql[<i class='fa fa-database'></i> PostgreSQL DB]

    classDef external fill:#D1E8E2,stroke:#333,stroke-width:2px;
    classDef app fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class youtube,genai_api,oauth_provider,postgresql external;
    class routes,ui_components,summarization_workflow,auth_service,sse_service,localization_service,db_adapter,genai_adapter,transcript_provider,models app;
```
