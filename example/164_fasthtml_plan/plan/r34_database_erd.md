Of course. Here is the text section for the Database Schema, including the Entity Relationship Diagram (ERD).

***

### 4. Database Schema (Entity Relationship Diagram)

An Entity Relationship Diagram (ERD) is a crucial tool for visualizing the logical structure of a database. This diagram represents the formal schema for RocketRecap v2, as detailed in the "Database Layout Proposal." It moves beyond the simple, single-table prototype to a normalized and scalable PostgreSQL design.

The core of the schema consists of two primary tables: `users` and `summary_jobs`. By separating these concerns, we establish a clear ownership model for data, which is essential for a multi-user application. The diagram clearly illustrates the **one-to-many relationship** between these tables: a single user can create many summarization jobs, but each job belongs to exactly one user. This relationship is enforced by a foreign key constraint. The schema also explicitly defines the custom `job_status` ENUM type to manage the job lifecycle robustly.

```mermaid
erDiagram
    users {
        BIGINT id PK "Auto-incrementing primary key"
        TEXT email UK "Unique user email address"
        TEXT oauth_provider "e.g., 'google'"
        TEXT oauth_provider_id "User's unique ID from the provider"
        VARCHAR(10) preferred_language "User's default language"
        TIMESTAMPTZ created_at "Timestamp of user creation"
        TIMESTAMPTZ updated_at "Timestamp of last update"
    }

    summary_jobs {
        BIGINT id PK "Auto-incrementing primary key"
        BIGINT user_id FK "References users.id"
        job_status status "ENUM type for job state"
        TEXT source_url "Original YouTube URL"
        TEXT model_id "e.g., 'gemini-1.5-pro-latest'"
        NUMERIC(10,8) cost_total_usd "Sum of all cost components"
        VECTOR(768) summary_embedding "(Optional) For similarity search"
        TIMESTAMPTZ created_at "Timestamp of job creation"
        TIMESTAMPTZ completed_at "Timestamp of completion or failure"
    }

    users ||--o{ summary_jobs : "has"

```
