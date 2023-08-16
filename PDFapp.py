from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, openai


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF")
    # uploading PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    # extracting the text from PDF file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Creating Embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            #answering users question
            user_question = st.text_input("Ask a question about your PDF :")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            response: str = chain.run(input_documents=docs, question=user_question)

            st.write(response)






if __name__ == '__main__':
    main()

