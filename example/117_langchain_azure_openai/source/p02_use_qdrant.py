#|default_exp p02_use_qdrant
# python -m venv ~/llm_env; . ~/llm_env/bin/activate; source ~/llm_environment.sh;
# pip install qdrant-client langchain[llms] openai sentence-transformers
# deactivate
import os
import time
import numpy as np
import langchain
import qdrant_client
import openai
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
start_time=time.time()
debug=True
_code_git_version="b14b86a907a19c83d19b66669c9f94e4539b2328"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/117_langchain_azure_openai/source/"
_code_generation_time="21:55:48 of Tuesday, 2023-09-12 (GMT+1)"
langchain.debug=True
chatgpt_deployment_name="gpt-35"
chatgpt_model_name="gpt-35-turbo"
openai.api_type="azure"
openai.api_key=os.getenv("OPENAI_API_KEY")
openai.api_base=os.getenv("OPENAI_API_BASE")
openai.api_version=os.getenv("OPENAI_API_VERSION")
llm=AzureChatOpenAI(temperature=0, model_name=chatgpt_model_name, deployment_name=chatgpt_deployment_name)
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-6-v3")
client=QdrantClient(host="localhost", port=6333, prefer_grpc=False)
COLLECTION_NAME="aiw"
qdrant=Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings, metadata_payload_key="payload")
retriever=qdrant.as_retriever()
qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
question="What does Alice drink?"
answer=qa.run(question)
print(answer)