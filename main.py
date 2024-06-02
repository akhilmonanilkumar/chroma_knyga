import os
import uuid
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from chat_storage import RedisChatManager

# Load environment variables from .env file
load_dotenv('.env')

# Set the OpenAI API key from the environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Set the REDIS key from the environment variables
REDIS_KEY = os.environ['REDIS_KEY']


# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=3000)

# Initialize OpenAI embeddings function
embedding_fn = OpenAIEmbeddings(model='text-embedding-3-large')

# Initialize the language model with specified parameters
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize Redis chat manager for storing and retrieving chat histories
redis = RedisChatManager(host="localhost", port=6379, db=0, password=REDIS_KEY)


# Function to create a new ChromaDB collection for a user
def create_collection(user_id):
    collection_name = f"{user_id}_{uuid.uuid4()}"
    chroma_client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"})
    return collection_name


# Function to list all ChromaDB collections
def list_collections():
    collections = chroma_client.list_collections()
    return collections


# Function to delete a specific ChromaDB collection by name
def delete_collection(collection_name: str):
    chroma_client.delete_collection(collection_name)
    return True


# Function to delete all ChromaDB collections
def delete_collections():
    collections = list_collections()
    for collection in collections:
        chroma_client.delete_collection(collection.name)
    return True


# Function to generate embeddings from a PDF document
def generate_embeddings_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])
    return chunks, embeddings


# Function to save embeddings from a PDF to a ChromaDB collection
def save_embeddings(user_id: str, book_id: str, pdf_path: str):
    collection_name = f"{user_id}_{uuid.uuid4()}"
    collection = chroma_client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"})
    chunks, embeddings = generate_embeddings_from_pdf(pdf_path)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        document_id = f"{book_id}_{i}"
        collection.upsert(
            ids=document_id,
            embeddings=embedding,
            metadatas={
                "book_id": book_id,
                "user_id": user_id,
                "source": chunk.metadata['source'],
                "page": chunk.metadata['page']},
            documents=chunk.page_content)
    print(collection_name)
    return collection_name


# Function to run a language model chain with character-specific context
def llm_characters_chain(query: str, collection_name: str):
    context = Chroma(client=chroma_client,
                     embedding_function=embedding_fn,
                     collection_name=collection_name,
                     collection_metadata={"hnsw:space": "cosine"})
    retriever = context.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }) | prompt | llm | output_parser
    response = chain.invoke({"question": query})
    print("Response : " + response)
    return response


# Function to run a language model chain to summarize information
def llm_summary_chain(query: str, collection_name: str):
    context = Chroma(client=chroma_client,
                     embedding_function=embedding_fn,
                     collection_name=collection_name,
                     collection_metadata={"hnsw:space": "cosine"})
    retriever = context.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }) | prompt | llm | output_parser
    response = chain.invoke({"question": query})
    print("Response : " + response)
    return response


# Function to run a language model chain for chat-based interaction
def llm_chat_chain(query: str, user_id: str, chat_id: str, collection_name: str):
    context = Chroma(client=chroma_client,
                     embedding_function=embedding_fn,
                     collection_name=collection_name,
                     collection_metadata={"hnsw:space": "cosine"})
    history = redis.get_chat(user_id=user_id, chat_id=chat_id)
    template = """Answer the question based only on the following context and history:
    {context} {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    retriever = context.as_retriever()
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: history
    }) | prompt | llm | output_parser
    response = chain.invoke({"question": query})
    redis.save_chat(user_id=user_id, chat_id=chat_id, query=query, response=response)
    print("Response : \n" + response)
    return response


if __name__ == "__main__":
    # llm_summary_chain(query="summarise", collection_name="user_001_collection")
    # llm_characters_chain(query="list the main characters and their role", collection_name="user_001_collection")
    # llm_chat_chain(query="what is the context of the story?", user_id="001", chat_id="001", collection_name="user_001_collection")
    # redis.clear_database()
    # redis.user_chats(user_id="001")
    # delete_collections()
    # get_collections()
    save_embeddings(user_id="001", book_id="001", pdf_path="data/romeo-and-juliet.pdf")
