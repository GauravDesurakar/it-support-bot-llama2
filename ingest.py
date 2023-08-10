"""
RecursiveCharacterTextSplitter: This class is likely a custom implementation that splits text documents into smaller chunks to prepare them for embedding.
PyPDFLoader: A class used to load PDF documents.
DirectoryLoader: A class used to load documents from a directory.
HuggingFaceEmbeddings: This class is used to generate text embeddings using models from Hugging Face's Transformers library.
FAISS: A class used for managing and performing similarity search on high-dimensional vectors.
"""

# Import necessary classes from modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

"""
Defining Paths:
DATA_PATH: This constant specifies the path to the directory containing the PDF documents.
VECTOR_DB_PATH: This constant specifies the path where the FAISS vector database will be saved.
"""
# Define paths for data and vector database
DATA_PATH = "data/"
VECTOR_DB_PATH = "vectorstores/db_faiss"


# Define a function to build the vector database
def func_build_vector_db():
    # Initialize a directory loader to load PDF documents from the specified path
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    # Load the documents using the loader
    documents = loader.load()
    # Initialize a text splitter to split documents into smaller chunks to prepare them for embedding.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # Split the documents into chunks of text
    texts = text_splitter.split_documents(documents)

    # Initialize HuggingFace embeddings using the 'sentence-transformers/all-MiniLM-L6-v2' model from Hugging Face's Transformers library.
    # The embeddings are computed on CPU (as specified by model_kwargs).
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create vectors from the text chunks using the embeddings and store them in the vector database (FAISS)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_PATH)


# Entry point of the script
if __name__ == '__main__':
    # Call the function to build the vector database
    func_build_vector_db()
