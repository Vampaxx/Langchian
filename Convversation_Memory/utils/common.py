import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
#from langchain.prompt import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



def get_pdf_file(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator   = "\n",
        chunk_size  = 1000, #numbers of charactors
        chunk_overlap  = 200, # protect the meaning of paragraph, if the chunk is overlap
        length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

