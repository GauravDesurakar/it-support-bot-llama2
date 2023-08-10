from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from loguru import logger


VECTOR_DB_PATH = "vectorstores/db_faiss"

custom_prompt_template = """
You are professional Computer Technician. You have expertise in maintaining computer systems, troubleshooting errors, and repairing the organization's hardware.
Use the following pieces of information to answer the user's question. Always try to provide answers in bullet point.
If you don't know the answer then please tell that you are not sure about the answer. 
Use provided source to answer the question. Don't try to make up the answer.

Context: {context}
Question: {question}

"""


# Load the LLM model
def func_load_llm():
    """
    :return:
        Returns the created llm instance of the LangLink Model from the function.
    """
    logger.debug("Into def func_load_llm")
    llm = CTransformers(
        model="llm_model/llama-2-7b.ggmlv3.q8_0.bin", # Specifies the path to the pre-trained LangLink Model (LLM) binary file to be loaded
        model_type="llama", # indicating the type of the LangLink Model being loaded.
        max_new_tokens=512, # maximum number of new tokens to be generated. It controls length of response.
        temperature=0.6
    )
    return llm


# Define function to set custom prompt
def func_set_custom_prompt():
    """
    PromptTemplate: Prompts are input texts or instructions given to a language model to guide its text generation.
    The PromptTemplate class provides a way to define the structure of prompts using placeholders like {context} and {question}.

    :return:
        Returns the created PromptTemplate instance from the function.
    """
    logger.debug("Into def func_retrieval_qa_chain")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    logger.info(f"prompt: {prompt}")
    return prompt


# Define a function to RetrievalQA chain using the specified LLM, prompt, and vector store (db).
def func_retrieval_qa_chain(llm, prompt, db):
    """
    llm:  LangLink Model (LLM) instance, used for text generation.
    prompt: A structured prompt template to guide the generation process.
    db: Vector store (database) instance, used for retrieval.
    return:
        Returns the configured RetrievalQA chain.
    """
    logger.debug("Into def func_retrieval_qa_chain")
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff', # The chain type is set as 'stuff' for retrieval QA chain.
                                           retriever=db.as_retriever(search_kwargs={'k': 1}), # no.of results
                                           return_source_documents=True,# Ensures response from provided Document
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# QA Model Function
def func_qa_bot():
    logger.debug("Into def func_qa_bot")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", # generating embeddings
                                       model_kwargs={'device': 'cpu'}) # To model should be loaded on the CPU
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings) # Loads a local FAISS vector store
    llm = func_load_llm() # Calling definition func_load_llm
    qa_prompt = func_set_custom_prompt()  # Calling definition func_set_custom_prompt
    qa = func_retrieval_qa_chain(llm, qa_prompt, db) # Calling definition func_retrieval_qa_chain
    logger.info(f"func_final_result: {qa}")
    return qa


# output function
def func_final_result(query):
    logger.debug("Into def func_final_result")
    qa_result = func_qa_bot()  # Calling definition func_qa_bot
    response = qa_result({'query': query})
    logger.info(f"func_final_result: {response}")
    return response

