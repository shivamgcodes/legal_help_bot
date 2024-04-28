from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.chains import retrieval_qa
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_community.llms import Ollama
from langchain.chains import VectorDBQA , retrieval_qa , RetrievalQA
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import streamlit as st
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.prompts import (ChatPromptTemplate, FewShotChatMessagePromptTemplate)

# read the files
pdffiles = [ r"C:\Users\hp\Desktop\legal hackathon chatbot\constitution of india.pdf",
               r"C:\Users\hp\Desktop\legal hackathon chatbot\ipc.pdf"]

reader = PdfReader(pdffiles[0]) 
# printing number of pages in pdf file 

print(len(reader.pages))   
# creating a page object 
embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = vectordb = Chroma.from_texts(" ", embedding = embedding_func)

#putting the files into the vector database

for file in pdffiles:
    reader = PdfReader(file) 
    for i in range(len(reader.pages)):
        page = reader.pages[i] 
        documents = [page.extract_text(),]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(documents[0])

        vectordb.aadd_texts(texts = texts, embedding = embedding_func)


#prompts 

promptqa = hub.pull("rlm/rag-prompt" , api_url="https://api.hub.langchain.com")
print(type(promptqa))

promptqa_mod = ChatPromptTemplate.from_messages(
  ("human", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. keep the answer concise . answer should be in the format of four points , each point will first name the rule or the IPC sections (like section 302 , 129 or whatever applies) and then explain it briefly  \n Question: {question}\n Context: {context} \n Answer:"),
)
promptqa_try = ChatPromptTemplate.from_messages(
  ("human", "return me back the question and the context AS IT IS  \n Question: {question}\n Context: {context} \n Answer:"),
)


retriever = vectordb.as_retriever()
llm=Ollama(model="mistral")
retrieval_chain = RetrievalQA.from_llm(llm = llm , prompt = promptqa_mod, retriever=retriever  )

prompt_template = "answer the question like a very intelligent indian advocate  - {question} . give me the answer in four different points , each point strictly seperated "
prompt_hyde = PromptTemplate(
    input_variables=["question"], template=prompt_template
)
hyde_chain =  LLMChain(llm=llm, prompt=prompt_hyde , verbose = False )

st.title('Legal assistant')
input_text=st.text_input("ask your question")
output = hyde_chain.invoke(input_text )

# print(type(output))
# print(output.keys())
# print(output)

output = retrieval_chain.invoke(output["text"] )


st.write(output['result'])