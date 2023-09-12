| file  | level | content         |
|-------|-------|-----------------|
| gen00 | 30%   | simple example  |
| gen01 |       | fill index      |
| gen02 |       | search in index |
|       |       |                 |

- general introduction to langchain
- https://www.youtube.com/watch?v=jfJbaJHnnP0 LangChain for beginners | full code
- https://github.com/RGGH/LangChain-Course
- https://python.langchain.com/docs/integrations/vectorstores/qdrant
- how to use openai with azure:
https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Chat/basic_chatcompletions_example_sdk.ipynb
```
python -m venv ~/llm_env
. ~/llm_env/bin/activate
pip install langchain[llms] openai

# this installs 620MB torch-2.0.1 (why?)
# 560MB of cuda runtime? i don't have nvidia hw
# finally llm_env is 4.6GB
```

```
langchain-0.0.285
openai-0.28.0
```

- authentication

```
https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python
```
 
- api key, endpoint, deployment name. all of these can be obtained
  from inside Azure AI Studio (oai.azure.com/portal)
- i called my deployments gpt-35, gpt-35-16k, gpt-4, gpt-4-32k, gpt4, gpt35, gpt4_32k
- to see api key and endpoint go to settings (gear icon) -> Resource
- Endpoint: https://<resource-name>.openai.azure.com/
- Key: 5929bf3.... (only one key is shown, note: this is not even part of a real key)

- if i go to portal.azure.com, click on the resource and `Keys and
  Endpoints` then i see two keys, KEY 1 is the one that is shown in
  the Azure AI Studio

```
echo export AZURE_OPENAI_KEY="REPLACE_WITH_YOUR_KEY_VALUE_HERE" >> /etc/environment && source /etc/environment
echo export AZURE_OPENAI_ENDPOINT="REPLACE_WITH_YOUR_ENDPOINT_HERE" >> /etc/environment && source /etc/environment
export OPENAI_API_KEY="..."

export OPENAI_API_TYPE=azure
export OPENAI_API_VERSION=2023-05-15
export OPENAI_API_BASE=https://<resource-name>.openai.azure.com
export OPENAI_API_KEY="<key-as-hexadecimal-string>"

```
- /etc/environment is used for something else in gentoo
- i will use ~/llm_environment.sh
```
source ~/llm_environment.sh 
. ~/llm_env/bin/activate

```


```

with get_openai_callback() as cb

cb.total_tokens
cb.prompt_tokens
cb.completion_tokens
cb.total_cost

```
# List packages in virtual env

```
pip freeze --local
aiohttp==3.8.5
aiosignal==1.3.1

```

# Example call

```
(llm_env) martin@archlinux ~/stage/cl-py-generator/example/117_langchain_azure_openai/source $ python -i p02_use_qdrant.py 
[chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
{
  "query": "How does a COVID infection typically progress?"
}
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] Entering Chain run with input:
[inputs]
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
{
  "question": "How does a COVID infection typically progress?",
  "context": "Manchester UK. He sends a link to a BBC article about lung damage and wants to hear Dr. Griffin's\n\n34:37.840 --> 34:44.320\n thoughts. This is an article about a new scanning mechanism that has identified lung damage months\n\n34:44.320 --> 34:52.240\n after the initial infection. I think about this. Yeah, you know, I actually appreciate that you\n\n34:52.240 --> 35:00.000\n bring this up, Paul. You know, if anything, I think people underestimate the damage, the morbidity\n\n35:00.000 --> 35:05.200\n of COVID-19, you know, some less than wise people have sort of said, Oh, COVID either, you know,\n\n35:05.200 --> 35:13.280\n you get it and either live or you die. Yeah, no, most people who get COVID don't just live or die.\n\n35:13.280 --> 35:18.880\n A lot of the people that don't die are really negatively impacted. And we see the people with\n\n35:18.880 --> 35:25.600\n the long haul, the tale of COVID. But there's also a lot of individuals who were physically\n\n35:25.600 --> 35:31.360\n\nCOVID-19. And, you know, yesterday, the numbers were almost 3000 today, we have over 3000 people\n\n03:18.080 --> 03:23.920\n who died here in the US of COVID-19. We've reached capacity at the hospitals here in New York.\n\n03:25.040 --> 03:28.640\n You know, I got a alert from one of the hospitals yesterday, you know, we have, you know,\n\n03:29.360 --> 03:34.560\n numbers of people stuck here in the ER because there's no open beds. The ICUs are overflowing\n\n03:34.560 --> 03:40.880\n into the other areas. This is not a great sign. I mean, we are not even seeing the post Thanksgiving\n\n03:40.880 --> 03:47.360\n search. So just want people to sort of think about when they're making their decisions. We are as\n\n03:47.360 --> 03:52.400\n bad right now as it has ever gotten. More people died today than have ever died in a single day\n\n03:52.400 --> 03:58.080\n from COVID-19 here in the States. More people are in the hospital right now with COVID-19 than ever.\n\n03:58.080 --> 04:05.040\n\nactive and they find themselves compromised. We've had several several serious athletes who find\n\n35:31.360 --> 35:35.120\n that, you know, they went from running six, seven miles a day to having trouble going up the flight\n\n35:35.120 --> 35:40.320\n of stairs. We're appreciating that there can be for a lot of individuals much more lung damage\n\n35:40.320 --> 35:46.720\n than was initially appreciated. We're also recognizing heart damage as well. So I think these studies\n\n35:46.720 --> 35:52.400\n are really important to let people know that you really want to not get COVID. Because yeah,\n\n35:52.400 --> 35:56.480\n you may survive, you may come out the other side, you may be able to tell everyone how great you're\n\n35:56.480 --> 36:01.760\n doing. But let's see you try to run a mile after you've had COVID. It may not go so well. So\n\n36:02.720 --> 36:08.800\n let's really for the next few months until, and I guess this reminds me of my other movie as we'll\n\n36:08.800 --> 36:13.440\n\nsymptoms, but this is going to be moving hopefully to better therapeutics. But there are thousands\n\n30:51.680 --> 30:58.400\n and hundreds of thousands of individuals suffering with this post COVID syndrome.\n\n30:58.400 --> 31:04.720\n And we're committed to finding answers. All right. Next one is from Chaim who lives in\n\n31:04.720 --> 31:09.600\n Washington, D.C. And he says, my kid's school is doing almost everything right, distancing,\n\n31:09.600 --> 31:14.960\n mask wearing, you know, no moving around, no testing, unfortunate. He says there's no testing\n\n31:14.960 --> 31:22.080\n as in addition to practical issues, they seem to be worried about moral hazard. Anyway, new cases\n\n31:22.080 --> 31:26.960\n are spiking here in D.C. We are now cresting the record set at the peak in May. Mayor has tightened\n\n31:26.960 --> 31:32.880\n some restrictions, but indoor dining still permitted. So I think things are going to get worse. His\n\n31:32.880 --> 31:37.760"
}
[llm/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:AzureChatOpenAI] Entering LLM run with input:
{
  "prompts": [
    "System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\nManchester UK. He sends a link to a BBC article about lung damage and wants to hear Dr. Griffin's\n\n34:37.840 --> 34:44.320\n thoughts. This is an article about a new scanning mechanism that has identified lung damage months\n\n34:44.320 --> 34:52.240\n after the initial infection. I think about this. Yeah, you know, I actually appreciate that you\n\n34:52.240 --> 35:00.000\n bring this up, Paul. You know, if anything, I think people underestimate the damage, the morbidity\n\n35:00.000 --> 35:05.200\n of COVID-19, you know, some less than wise people have sort of said, Oh, COVID either, you know,\n\n35:05.200 --> 35:13.280\n you get it and either live or you die. Yeah, no, most people who get COVID don't just live or die.\n\n35:13.280 --> 35:18.880\n A lot of the people that don't die are really negatively impacted. And we see the people with\n\n35:18.880 --> 35:25.600\n the long haul, the tale of COVID. But there's also a lot of individuals who were physically\n\n35:25.600 --> 35:31.360\n\nCOVID-19. And, you know, yesterday, the numbers were almost 3000 today, we have over 3000 people\n\n03:18.080 --> 03:23.920\n who died here in the US of COVID-19. We've reached capacity at the hospitals here in New York.\n\n03:25.040 --> 03:28.640\n You know, I got a alert from one of the hospitals yesterday, you know, we have, you know,\n\n03:29.360 --> 03:34.560\n numbers of people stuck here in the ER because there's no open beds. The ICUs are overflowing\n\n03:34.560 --> 03:40.880\n into the other areas. This is not a great sign. I mean, we are not even seeing the post Thanksgiving\n\n03:40.880 --> 03:47.360\n search. So just want people to sort of think about when they're making their decisions. We are as\n\n03:47.360 --> 03:52.400\n bad right now as it has ever gotten. More people died today than have ever died in a single day\n\n03:52.400 --> 03:58.080\n from COVID-19 here in the States. More people are in the hospital right now with COVID-19 than ever.\n\n03:58.080 --> 04:05.040\n\nactive and they find themselves compromised. We've had several several serious athletes who find\n\n35:31.360 --> 35:35.120\n that, you know, they went from running six, seven miles a day to having trouble going up the flight\n\n35:35.120 --> 35:40.320\n of stairs. We're appreciating that there can be for a lot of individuals much more lung damage\n\n35:40.320 --> 35:46.720\n than was initially appreciated. We're also recognizing heart damage as well. So I think these studies\n\n35:46.720 --> 35:52.400\n are really important to let people know that you really want to not get COVID. Because yeah,\n\n35:52.400 --> 35:56.480\n you may survive, you may come out the other side, you may be able to tell everyone how great you're\n\n35:56.480 --> 36:01.760\n doing. But let's see you try to run a mile after you've had COVID. It may not go so well. So\n\n36:02.720 --> 36:08.800\n let's really for the next few months until, and I guess this reminds me of my other movie as we'll\n\n36:08.800 --> 36:13.440\n\nsymptoms, but this is going to be moving hopefully to better therapeutics. But there are thousands\n\n30:51.680 --> 30:58.400\n and hundreds of thousands of individuals suffering with this post COVID syndrome.\n\n30:58.400 --> 31:04.720\n And we're committed to finding answers. All right. Next one is from Chaim who lives in\n\n31:04.720 --> 31:09.600\n Washington, D.C. And he says, my kid's school is doing almost everything right, distancing,\n\n31:09.600 --> 31:14.960\n mask wearing, you know, no moving around, no testing, unfortunate. He says there's no testing\n\n31:14.960 --> 31:22.080\n as in addition to practical issues, they seem to be worried about moral hazard. Anyway, new cases\n\n31:22.080 --> 31:26.960\n are spiking here in D.C. We are now cresting the record set at the peak in May. Mayor has tightened\n\n31:26.960 --> 31:32.880\n some restrictions, but indoor dining still permitted. So I think things are going to get worse. His\n\n31:32.880 --> 31:37.760\nHuman: How does a COVID infection typically progress?"
  ]
}
[llm/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:AzureChatOpenAI] [2.99s] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection.",
        "generation_info": {
          "finish_reason": "stop"
        },
        "message": {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection.",
            "additional_kwargs": {}
          }
        }
      }
    ]
  ],
  "llm_output": {
    "token_usage": {
      "completion_tokens": 94,
      "prompt_tokens": 1181,
      "total_tokens": 1275
    },
    "model_name": "gpt-35-turbo"
  },
  "run": null
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] [2.99s] Exiting Chain run with output:
{
  "text": "A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection."
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] [2.99s] Exiting Chain run with output:
{
  "output_text": "A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection."
}
[chain/end] [1:chain:RetrievalQA] [3.02s] Exiting Chain run with output:
{
  "result": "A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection."
}
A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection.
>>> 
```
