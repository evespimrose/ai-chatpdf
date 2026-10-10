from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import init_chat_model

from langsmith import Client

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
import os

from langchain_chroma import Chroma

def format_docs(docs):
         return '\n\n'.join(doc.page_content for doc in docs)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = Client()
prompt = client.pull_prompt('rlm/rag-prompt',dangerously_pull_public_prompt=True)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key= api_key)

loader = PyPDFLoader('dong.pdf')
pages = loader.load_and_split()

#텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)
texts = text_splitter.split_documents(pages)

db = Chroma.from_documents(texts, embeddings_model)

llm = init_chat_model(
    "gpt-5.6-luna",  # gpt-5.6-luna, gpt-4o, gpt-4o-mini 등 유효한 모델명 사용
    # Kwargs passed to the model:
    timeout=120,
    max_tokens=1000,
    max_retries=10,  # Default; increase for unreliable networks
    api_key = api_key,
)

retriever_llm = MultiQueryRetriever.from_llm(
  retriever=db.as_retriever(search_kwargs={"k": 3}),
  llm=llm,
)

#사용자 질문에 대한 정보 가져오기
question = "주인공은 점순이를 어떻게 생각해? 여성으로써 좋아하나요?"
docs = retriever_llm.get_relevant_documents(question)
#print(format_docs(docs))

client = Client()
prompt = client.pull_prompt('rlm/rag-prompt',dangerously_pull_public_prompt=True)

def format_docs(docs):
         return '\n\n'.join(doc.page_content for doc in docs)

rag_chain = (
  {'context':retriever_llm | format_docs, "question":RunnablePassthrough()}  #입력값 그대로 사용
  | prompt
  | llm
  | StrOutputParser()
)

result = rag_chain.invoke(question)
print(result)