
- https://www.youtube.com/watch?v=jfJbaJHnnP0 LangChain for beginners | full code

```
python -m venv ~/llm_env
. ~/llm_env/bin/activate
pip install langchain openai

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
- i called my deployments gpt-35, or gpt-35-16k
- to see api key and endpoint go to settings (gear icon) -> Resource
- Endpoint: https://<resource-name>.openai.azure.com/
- Key: 5929bf3.... (only one key is shown)

- if i go to portal.azure.com, click on the resource and `Keys and
  Endpoints` then i see two keys, KEY 1 is the one that is shown in
  the Azure AI Studio