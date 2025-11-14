| file  | description                                                         |
|-------|---------------------------------------------------------------------|
| gen01 | simple test                                                         |
| gen02 | tool use (reading directory listings, files and searching for text) |
|       |                                                                     |


### **Example 168: Analyzing a Codebase with a Gemini Agent**

This example was created for the homework in Lecture 2 of Jeremy Howard's 'How To Solve It With Code' course (2025). It demonstrates the lecture's concepts by using an LLM agent with filesystem tools to analyze a codebase, adapted here to use Google's Gemini model via the `lisette` library.

#### **Motivation**

I'm really interested in using LLMs to help understand legacy codebases. This experiment is a practical attempt to do just that: point an agent at an old project of mine and see if it can produce a useful summary.

#### **Implementation**

*   **Agent:** This code uses `lisette` instead of `claudette`, primarily because I have a free and generous Gemini API key. The switch between the two libraries is straightforward.
*   **Tools:** As described in the lecture, the agent is given access to filesystem tools (`rg`, `set` and `view` from `fastcore.tools`) to list directories and read files.
*   **Target:** The agent was run on one of my old Git repositories (this one), a place where I occasionally add new experiments.

The full script can be found in `p02_fs_tools.py`.

#### **Results and Observations**

The agent produced the overview found in `output2.md`. I was impressed; it's a nice summary of the project structure.

However, it can be quite scary to watch it spool through so many tool requests. This highlights a critical problem with autonomous agents: without supervision, they can quickly become expensive and unpredictable. I need to look into the debugging facilities to observe and control the prompts before they are submitted.

#### **A Follow-Up Idea: Training for Software Architecture**

This experiment sparked a thought. Jeremy mentioned in the lecture that LLMs are bad at software architecture because there isn't enough training data.

Perhaps we could generate such data. What if we scraped public issue trackers (Jira, GitHub Issues) and connected tickets to their corresponding pull requests? This could teach an AI how a software project actually grows—how a feature request becomes code, and what kinds of changes lead to bugs later on. It seems like a scalable way to teach an AI how software is really built.
