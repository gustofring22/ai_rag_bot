import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
from requests.exceptions import HTTPError

#Step 1: Setup llm (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(repo_id=huggingface_repo_id,
                              temperature=0.5,
                              model_kwargs={
                                  "token": HF_TOKEN,
                                    "max_length": 512,
                              })
    return llm


#Step 2: Connect LLM with FAISS and Create Chain
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, you can say "I don't know". Do not try to make up an answer.
Do not provide anything out of the given context.

Context: {context}
Quetion: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


#Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#Create Chain
qa_chain = RetrievalQA.from_chain_type(llm=load_llm(huggingface_repo_id),
                            chain_type='stuff',
                            retriever=db.as_retriever(search_kwargs={'k':3}),
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},
)           


user_query = input("Ask me a question: ")
response = qa_chain.invoke({'query': user_query})
print('result:', response['result'])
print('source_documents:', response['source_documents'])