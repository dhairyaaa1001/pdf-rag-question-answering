# pdf_qa_engine.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate

class PDFQuestionAnsweringEngine:
    def __init__(self, pdf_path):
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError("File not found! Please check the path.")

        loader = PyPDFLoader(pdf_path)
        all_pages = loader.load()

        if len(all_pages) > 1000:
            raise ValueError("PDF is too long! Please upload a PDF with 1000 pages or fewer.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_pages)

        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            max_length=1024,
            temperature=0,
            no_repeat_ngram_size=3
        )
        self.llm = HuggingFacePipeline(pipeline=llm_pipeline)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                Answer the question based on the context below. 
                If the answer isn't in the context, say you don't know.

                Context: {context}

                Question: {question}

                Answer:
            """
        )

    def ask(self, question):
        retrieved_docs = self.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = self.prompt_template.format(context=context_text, question=question)
        return self.llm.invoke(final_prompt)
