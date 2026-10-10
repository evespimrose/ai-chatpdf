__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain.chat_models import init_chat_model
import streamlit as st
import os
import tempfile

from langsmith import Client

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

api_key = st.text_input("OPENAI_API_KEY", type="password")

st.title("ChatPDF 📒")
st.write('---')

uploaded_file = st.file_uploader("PDF 파일을 올려주세요",type="pdf")
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() # 임시폴더 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된 pdf로딩
    pages = loader.load_and_split()
    return pages
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # 분할
    text_split = RecursiveCharacterTextSplitter(
      chunk_size=100, #각 청크의 최대 길이
      chunk_overlap=20, #인접한 청크사이의 중복영역, 문장이 끊기는 문제 해결 20글자 겹침
      length_function=len,#청크길이 측정하는 함수
      is_separator_regex=False,#단순한 문자열로 해석
    )

    texts = text_split.split_documents(pages)
    # 임베딩후 벡터디비에 저장
    embeddings = OpenAIEmbeddings(
      model="text-embedding-3-large",
      api_key=api_key
    )

    # 벡터데이터베이스 설정 및 저장
    db = Chroma.from_documents(texts, embeddings)

    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # 스트리밍 처리할 Handler 생성
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    # user input
    st.header('PDF에게 질문하세요!!')
    question = st.text_input("질문을 입력하세요")
    if st.button('질문하기'):
        with st.spinner('Wait for it....'):
            # 검색기능 구현
            
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
              #prompt Template hub
            client = Client()
            prompt = client.pull_prompt('rlm/rag-prompt',dangerously_pull_public_prompt=True)


            # 출력공간 확보 stream 부분
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
                
            generate_llm = init_chat_model(model="gpt-5.5", openai_api_key=api_key, streaming=True, callbacks=[stream_handler])

            def format_docs(docs):
                return '\n\n'.join(doc.page_content for doc in docs)

            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                llm=llm
            )

            rag_chain = (
                {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}
                | prompt
                | generate_llm  #스트리밍 시
                | StrOutputParser()
            )

            result = rag_chain.invoke(question)
    

    # 프롬프트, llm 체인 실행