| file  | description                                                         |
|-------|---------------------------------------------------------------------|
| gen01 | simple test                                                         |
| gen02 | tool use (reading directory listings, files and searching for text) |
|       |                                                                     |

Your notes are fine, but they can be more structured and critical. Here is a revised version.

---

### **Homework Notes: Lecture 2 of the 'How To Solve It With Code' course 2025 by Jeremy Howard**

**1. Experiment: Codebase Analysis with a Gemini-based Agent**

*   **Objective:** Evaluate an LLM agent's ability to comprehend a legacy codebase.
*   **Method:** Substituted the lecture's Claude-based agent (`claudette`) with a Gemini-based one (`lisette`). The agent was granted filesystem access (`fs_tools`) and directed to analyze a personal Git repository.
*   **Result:** The agent produced a high-level summary of the repository's structure and contents. The output is a reasonable first-pass analysis, but its utility for a developer already familiar with the codebase is probably low.

**2. Observation: Unconstrained Tool Use**

*   The agent's recursive tool use generated a cascade of API calls to explore the filesystem.
*   This behavior highlights a significant issue with autonomous agents: without strict constraints or a supervising loop, they can generate unpredictable and potentially expensive sequences of actions. The cost in tokens and latency is non-trivial.
*   The intuition to use debugging tools for observing and intervening in the prompt-generation process is correct. This is a mandatory capability for developing reliable agents. I don't care about agents that can't be controlled.

**3. Idea: Training Data for Software Architecture**

*   **Premise:** The lecture stated that LLMs are poor at software architecture due to a lack of suitable training data.
*   **Proposal:** Generate this data by scraping public issue trackers (Jira, GitHub Issues) and correlating tickets with their corresponding pull requests and code changes.
*   **Analysis:**
    *   **Plausibility:** The concept is sound. Linking high-level intent (a feature request) to a concrete implementation (a code diff) could teach a model about the *process* of software evolution.
    *   **Primary Challenge (Signal-to-Noise):** This is the main problem. Public issue trackers are incredibly noisy. The data would be dominated by poorly written bug reports, duplicate issues, abandoned requests, and trivial code changes ("fix typo"). Filtering for genuinely architectural changes would be extremely difficult.
    *   **Secondary Challenge (Causality):** The public record is an incomplete, lossy representation of the design process. Critical architectural decisions are often made in design documents, private meetings, or chat channels, not in the ticket itself. A model trained on this data would learn correlation, not the underlying design rationale. It would see *what* was done, but not *why*.
    *   **Conclusion:** This approach would likely train a model to be a proficient *code mechanic*—good at implementing narrowly defined features or fixing specific bugs. It would not train a *software architect*. The dataset lacks the high-level, multi-year strategic context that defines true architectural work. The premise that a lack of *data* is the core problem might be wrong; the problem might be that architecture is a process of reasoning under uncertainty, not pattern matching on text.
