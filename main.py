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

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=3000)

# Initialize OpenAI embeddings function
embedding_fn = OpenAIEmbeddings(model='text-embedding-3-large')

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Initialize Redis chat manager
redis = RedisChatManager(host="localhost", port=6379, db=0)


def create_collection(user_id):
    collection_name = f"{user_id}_{uuid.uuid4()}"
    chroma_client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"})
    return collection_name


def list_collections():
    collections = chroma_client.list_collections()
    return collections


def delete_collection(collection_name: str):
    chroma_client.delete_collection(collection_name)
    return True


def delete_collections():
    collections = list_collections()
    for collection in collections:
        chroma_client.delete_collection(collection.name)
    return True


def generate_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])
    return chunks, embeddings


def save_embeddings(user_id: str, book_id: str, pdf_path: str):
    collection_name = f"{user_id}_{uuid.uuid4()}"
    collection = chroma_client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"})
    chunks, embeddings = generate_embeddings(pdf_path)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        document_id = f"{book_id}_{i}"
        collection.add(
            ids=document_id,
            embeddings=embedding,
            metadatas={
                "book_id": book_id,
                "user_id": user_id,
                "source": chunk.metadata['source'],
                "page": chunk.metadata['page']},
            documents=chunk.page_content)
    return collection_name


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
    # redis.clear_database()
    # redis.user_chats(user_id="001")
    llm_summary_chain(query="summarise", collection_name="user_001_collection")
    # delete_collections()
    # get_collections()
    # save_embeddings(user_id="001", book_id="001", pdf_path="data/romeo-and-juliet.pdf")
