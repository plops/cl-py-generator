
- https://www.youtube.com/watch?v=jfJbaJHnnP0 LangChain for beginners | full code
- https://github.com/RGGH/LangChain-Course
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