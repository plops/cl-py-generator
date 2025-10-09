A## Frequently Asked Questions (FAQ) about the Video Summarizer & Map

### **About the Service & Technology**

**Q: What is the main purpose of this service?**
**A:** While the service excels at creating concise summaries from technical lectures and video abstracts, the primary goal is to create a **map of all the videos I watch**. This map feature aims to give you control over my content consumption, countering the experience of being passively fed content by recommendation algorithms. Ideally I want to show the local neighborhood for every new video summary so that you can see what other youtube videos exist.

**Q: Who pays for that?**
**A:** You see I wrote "I" in the previous answer because I mainly build this for myself. Google gives a free quota of requests every day (**50 Pro, 5000 Flash**) that will be shared among all users of this website.

**Q: How does the AI know the content of the video if I only provide a link?**
**A:** For YouTube links, the system attempts to automatically download the video's captions/transcript using tools like `yt-dlp`. Once the transcript is obtained, it is fed to the large language model (LLM) for summarization. For some Chinese videos or videos without a transcript, I sometimes download the audio channel and use `whisper.cpp` to create a transcript. I haven't added this functionality to the website yet, because I don't consider this very important.

**Q: What are the most interesting applications beyond simple video summaries?**
**A:** For technical lectures (in particular medical or biology) I sometimes ask the AI to "provide a glossary of the medical terms." This can be quite helpful. I often watch video podcasts on the MicrobeTV Youtube channel and add their references from the video description (scientific papers) to the prompt, which can give additional information. Sometimes I add all the youtube comments or the realtime chat history to the transcript. This can be useful if the host asked a question in a video that is later discussed and answered by viewers. I haven't integrated this into the website yet, because it is quite difficult to get this information from YouTube, but you can copy and paste this information together with the YouTube video transcript if you want to try yourself. The ability to map watched videos is also a key application. Beyond summarizing content, the underlying technology (especially using LLM embeddings for clustering) has potential applications in complex research fields like **immune system research, cancer studies, and single-nucleus RNA sequencing**. 

**Q: What makes this service particularly useful for long technical lectures (1-2 hours)?**
**A:** Creating a summary often takes considerably less time than watching the entire video, providing a fast way to grasp the abstract and key points. I find that for technically difficult lectures I sometimes miss crucial information and it is good to have summaries of timestamps. Furthermore, the newer `flash` model has become quite useful for summarizing up to one hour of video content since September 2025.

**Q: Why doesn't YouTube provide summaries?**
**A:** I sometimes see YouTube comments as an example, and I think if you buy YouTube Premium you can access their summaries as a feature. From the examples I have seen, they only give a short sketch about what the video is about, whereas I try to make the summaries self-containing. I think for non-premium users YouTube is disincentivised to give you good self-contained summaries because they want you to view the videos and ads.

### **Performance and Cost**

**Q: How much energy/cost does running these summaries require?**
**A:** The energy consumption/cost is tracked using the estimated price as a proxy. Summaries are now considerably cheaper than they were a year ago and might even consume less energy than watching the entire video, though this is difficult to confirm precisely.

**Q: How do the costs for different models compare?**
**A:** The pricing structure for the available models (as of the system configuration) is:
*   **Flash Lite:** $\sim\$0.10$ per million input tokens / $\sim\$0.40$ per million output tokens.
*   **Flash:** $\sim\$0.30$ per million input tokens / $\sim\$2.50$ per million output tokens.
*   **Pro:** $\sim\$1.25$ per million input tokens / $\sim\$10.00$ per million output tokens.
The grounding with Google Search (which I haven't implemented in the production version yet) costs a whopping **$35 per 1000 prompts**.

**Q: What is the difference between training and inference costs for LLMs?**
**A:** The initial training of the massive LLMs that exist today is an extremely costly and energy-intensive, one-time process. The cost associated with this website is for **inference**—the cost of running the *already trained* network to generate a summary for you—which is significantly lower. My feeling is that it is worth it if many users can profit from the summaries, videos are more easily discoverable, and AI can produce a benefit to society in this way that will hopefully offset the initial training cost. If automatic processes run AI requests with huge prompts every night and no human ever gets them to see, then even the inference energy costs can add up to form a net loss for society.

**Q: What are the current usage quotas?**
**A:** The service relies on the free developer contingent, which is currently limited to **50 Pro requests** and **5000 Flash requests**. Hitting these limits results in inconvenient service interruptions. Google doesn't provide and I haven't implemented a counter for the quota yet; it is quite annoying because I don't know when and how often the counter is being reset.

### **Limitations and Future Work**

**Q: Why are some generated summaries factually incorrect or confusing?**
**A:** There are two main causes for inaccuracies:
1.  **Outdated Knowledge:** The base LLM may have outdated world knowledge (e.g., not knowing current political figures, sometimes it refers to Trump as a 'hypothetical president')—this messes up some of the summaries.
2.  **Transcript Quality:** If a YouTube video has multiple speakers, the downloaded transcript currently always lacks speaker attribution, which can confuse the AI when generating the summary.

**Q: What improvements are planned for the future?**
**A:** Future improvements aim to make the summaries significantly better by allowing the model to **enable "thinking"** (more complex reasoning) and **"Grounding with Google Search"** to ensure more up-to-date and verifiable information. Additionally, I plan to implement summary **translation** (so that I can watch Spanish and Chinese YouTube videos, maybe also expand to Chinese websites like Bilibili), add a **glossary**, and **integrate information from YouTube video comments and realtime chat** into the summary (perhaps an expert responds to a question of the host or an error is corrected).

**Q: Is there a plan to commercialize the website?**
**A:** Currently, there is no plan for commercialization. This is just for me to learn and keep track of what AI (in particular Gemini) is capable of.



