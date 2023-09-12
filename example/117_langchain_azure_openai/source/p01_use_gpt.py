#|default_exp p01_use_gpt
# python -m venv ~/llm_env; . ~/llm_env/bin/activate; source ~/llm_environment.sh;
# pip install qdrant-client langchain[llms] openai sentence-transformers
# deactivate
import os
import time
import numpy as np
import openai
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
start_time=time.time()
debug=True
_code_git_version="793c630a1277af23460b1f6829e0884562f1a13a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/117_langchain_azure_openai/source/"
_code_generation_time="19:54:03 of Tuesday, 2023-09-12 (GMT+1)"
chatgpt_deployment_name="gpt-35"
chatgpt_model_name="gpt-35-turbo"
openai.api_type="azure"
openai.api_key=os.getenv("OPENAI_API_KEY")
openai.api_base=os.getenv("OPENAI_API_BASE")
openai.api_version=os.getenv("OPENAI_API_VERSION")
# cd ~/src; git clone --depth 1 https://github.com/RGGH/LangChain-Course
# cd ~/Downloads ; wget https://github.com/qdrant/qdrant/releases/download/v1.5.1/qdrant-x86_64-unknown-linux-gnu.tar.gz
# mkdir q; cd q; tar xaf ../q/qdrant*.tar.gz
# cd ~/Downloads/q; ./qdrant
COLLECTION_NAME="aiw"
TEXTS=["/home/martin/src/LangChain-Course/lc5_indexes/text/aiw.txt"]
vectors=[]
batch_size=512
batch=[]
model=SentenceTransformer("msmarco-MiniLM-L-6-v3")
client=QdrantClient(host="localhost", port=6333, prefer_grpc=False)
def make_collection(client, collection_name: str):
    client.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
def make_chunks(input_text: str):
    text_splitter=RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=20, length_function=len)
    with open(input_text) as f:
        alice=f.read()
    chunks=text_splitter.create_documents([alice])
    return chunks
texts=make_chunks(TEXTS[0])
def gen_vectors(texts, model, batch, batch_size, vectors):
    for part in tqdm(texts):
        batch.append(part.page_content)
        if ( ((batch_size)<=(len(batch))) ):
            vectors.append(model.encode(batch))
            batch=[]
    if ( ((0)<(len(batch))) ):
        vectors.append(model.encode(batch))
        batch=[]
    vectors=np.concatenate(vectors)
    payload=list([item for item in texts])
    vectors=[v.tolist() for v in vectors]
    return vectors, payload
fin_vectors, fin_payload=gen_vectors(texts=texts, model=model, batch=batch, batch_size=batch_size, vectors=vectors)
def upsert_to_qdrant(fin_vectors, fin_payload):
    collection_info=client.get_collection(collection_name=COLLECTION_NAME)
    client.upsert(collection_name=COLLECTION_NAME, points=[PointStruct(id=((collection_info.vectors_count)+(idx)), vector=vector, payload=fin_payload[idx]) for idx, vector in enumerate(fin_vectors)])
make_collection(client, COLLECTION_NAME)
upsert_to_qdrant(fin_vectors, fin_payload)