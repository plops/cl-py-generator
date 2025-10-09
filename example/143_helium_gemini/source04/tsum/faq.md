## Frequently Asked Questions (FAQ) about the Video Summarizer & Map

### **About the Service & Technology**

**Q: What is the main purpose of this service?**
**A:** While the service excels at creating concise summaries from technical lectures and video abstracts, the primary goal is to create a **map of all the videos I watch**, i.e. https://rocketrecap.com/exports/index.html. This map feature aims to give **me** control over **my** content consumption, countering the experience of being passively fed content by recommendation algorithms. Ideally, I want to show the local neighborhood for every new video summary so that I can see what other YouTube videos exist. Eventually, I would like to support user accounts on the website so that everyone can see their individual map. *(Note: This map is currently a manually generated UMAP plot based on the first 4,000 entries and needs to be regenerated now (2025-10) that the database has reached 8,000 entries.)*

**Q: Who pays for this?**
**A:** You see I wrote "I" in the previous answer because I mainly build this for myself. Google gives a free quota of requests every day (**50 Pro, 5000 Flash**) that will be shared among all visitors of this website.

**Q: How does the AI know the content of the video if I only provide a link?**
**A:** For YouTube links, the system attempts to automatically download the video's captions/transcript using tools like `yt-dlp`. Once the transcript is obtained, it is fed to the large language model (LLM) for summarization. For some Chinese videos or videos without a transcript, I sometimes download the audio channel and use `whisper.cpp` to create a transcript. I haven't added this functionality to the website yet, because I don't consider this very important. 

**Q: What are some interesting applications beyond simple video summaries?**
**A:** Beyond just summarizing, I use it for specific tasks: for lectures in technical fields like medicine or biology, I often ask the AI to **"provide a glossary of the medical terms,"** which is very helpful. For video podcasts, especially from the **MicrobeTV Youtube channel**, I often manually add scientific paper references from the video description into the prompt to get a more informed summary. I also sometimes include YouTube comments or the live chat history with the transcript—this is great if the host asks a question that is later discussed and answered by viewers. Integrating this extra data automatically is quite difficult right now, but you can copy and paste it yourself to try it out. On a larger scale, the underlying technology (especially using LLM embeddings for clustering) has potential applications in complex research fields like **immune system research, cancer studies, and single-nucleus RNA sequencing**. The ability to map watched videos is also a key application.

**Q: What makes this service useful for long technical lectures (1-2 hours)?**
**A:** Creating a summary often takes considerably less time than watching the entire video, providing a fast way to grasp the abstract and key points. I find that for technically difficult lectures I sometimes miss crucial information and it is good to have summaries with timestamps. Furthermore, the newer `flash` model has become quite useful for summarizing up to one hour of video content since September 2025.

**Q: Why doesn't YouTube provide detailed summaries?**
**A:** I sometimes see YouTube comments as an example, and I think if you buy YouTube Premium you can access their summaries as a feature. From the examples I have seen, they only give a short sketch about what the video is about, whereas I try to make the summaries self-containing. I think for non-premium users YouTube is disincentivised to give you good self-contained summaries because they want you to view the videos and ads.

### **Performance and Cost**

**Q: How much energy/cost does running these summaries require?**
**A:** The cost of generating a summary is used as a proxy for its energy consumption. While a precise comparison is difficult, generating a summary is now considerably cheaper than a year ago and may consume less energy than streaming an entire video. I am very interested in analyzing this but haven't found a good way to do so yet.

**Q: How do the costs for different models compare?**
**A:** Approximate pricing for the available models is as follows:
*   **Flash Lite:** $\sim\$0.10$ per million input tokens / $\sim\$0.40$ per million output tokens.
*   **Flash:** $\sim\$0.30$ per million input tokens / $\sim\$2.50$ per million output tokens.
*   **Pro:** $\sim\$1.25$ per million input tokens / $\sim\$10.00$ per million output tokens.
An optional grounding feature using Google Search, which is not enabled on the site, is significantly more expensive at around **$35 per 1,000 prompts** [cite: User Comment].

**Q: What is the difference between training and inference costs for LLMs?**
**A:** Training a large LLM is an extremely costly and energy-intensive, one-time process. This website only incurs **inference** costs—the much lower cost of running the *already trained* network to generate a summary for you. My feeling is that it is worth it if many users can profit from the summaries, videos are more easily discoverable, and AI can produce a benefit to society in this way that will hopefully offset the initial training cost. This works best when summaries are read by people, rather than being generated by automated processes that no one sees.

**Q: What are the current usage quotas?**
**A:** The service relies on a free developer quota, currently limited to **50 Pro requests** and **5000 Flash requests**. Hitting these limits can cause inconvenient service interruptions. Google doesn't provide and I haven't implemented a counter for the quota yet; it is quite annoying because I don't know when and how often the counter is being reset. [cite: User Comment]2

### **Limitations and Future Work**

**Q: Why are some generated summaries factually incorrect or confusing?**
**A:** There are two main causes for inaccuracies:
1.  **Outdated Knowledge:** The base LLM's knowledge may be outdated (e.g., it might not be aware of recent events or figures; sometimes it refers to Trump as a 'hypothetical president')—this messes up some of the summaries. [cite: User Comment]
2.  **Transcript Quality:** If a YouTube video has multiple speakers, the downloaded transcript currently always lacks speaker attribution, which can confuse the AI when generating the summary.

**Q: What improvements are planned for the future?**
**A:** Planned enhancements include:
*   Optional grounding with Google Search for more up-to-date and verifiable information.
*   Automatic **translation** of summaries (so that I can watch Spanish and Chinese YouTube videos, maybe also expand to Chinese websites like Bilibili).
*   A dedicated feature to generate a **glossary** from a video (which is currently possible via specific prompting).
*   Integration of YouTube comments and live chat to enrich summaries with audience interaction. I am particularly interested in highlighting errors that the host made or highlighting particularly insightful comments.

**Q: Is there a plan to commercialize the website?**
**A:** No, there are currently no plans for commercialization. This is just for me to learn and keep track of what AI (in particular Gemini) is capable of.
