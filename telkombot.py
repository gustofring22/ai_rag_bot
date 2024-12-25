import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import time
from requests.exceptions import HTTPError


load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
@st.cache_resource
def get_vectorstore():
    embedded_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedded_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Hugging Face token is missing or invalid.")

    print(f"Using Hugging Face Token: {HF_TOKEN[:8]}...")  # Debugging
    
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512,
        }
    )
    return llm


def load_llm_with_retry(huggingface_repo_id, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return load_llm(huggingface_repo_id, HF_TOKEN)
        except HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Failed to load LLM after multiple retries.")

def main():
    st.title('Telkom Bot')
    st.write('Welcome to Telkom Bot! Please enter your question below.')
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Ask me a question:')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
        custom_prompt_template = """
                Use the pieces of information provided in the context to answer user's question.
                If you don't know the answer, you can say "I don't know". Do not try to make up an answer.
                Do not provide anything out of the given context.

                Context: {context}
                Quetion: {question}

                Start the answer directly. No small talk please.
                """
        
        llm= load_llm_with_retry(huggingface_repo_id)
        
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error('Vectorstore not found')

            qa_chain = RetrievalQA.from_chain_type(llm= llm,
                            chain_type='stuff',
                            retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},
)   
            response = qa_chain.invoke({'query': prompt})
        
            result = response['result']
          
        
        
            
            
            st.session_state.chat_messages.append({'role': 'assistant', 'content':result})

        except Exception as e:
            st.error(f'An error occurred: {e}')

        


if __name__ == '__main__':
    main()