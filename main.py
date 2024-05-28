import os
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from chat_storage import RedisChatManager

# Load environment variables from a .env file
load_dotenv('.env')

# Set the OpenAI API key from the environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=3000)

# Initialize OpenAI embeddings function
embedding_fn = OpenAIEmbeddings(model='text-embedding-3-large')

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize Redis chat manager
redis = RedisChatManager(host="localhost", port=6379, db=0)


# Create a collection for a user in ChromaDB
def create_collection(user_id):
    collection_name = f"user_{user_id}_collection"
    chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return collection_name


# List all collections in ChromaDB
def get_collections():
    try:
        collections = chroma_client.list_collections()
        print(collections)
        return collections
    except Exception as e:
        print(f"An error occurred: {e}")


# Delete a specific user's collection in ChromaDB
def delete_collection(user_id):
    try:
        collection_name = f"user_{user_id}_collection"
        chroma_client.delete_collection(collection_name)
        print("Collection deleted successfully.")
        return {f"{collection_name} deleted successfully."}
    except Exception as e:
        print(f"An error occurred: {e}")


# Delete all collections in ChromaDB
def delete_collections():
    try:
        collections = get_collections()
        for collection in collections:
            print(collection.name)
            chroma_client.delete_collection(collection.name)
        print("Collections deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_relevant_documents(query, user_id, book_id):
    query_embedding = embedding_fn.embed_query(query)
    collection_name = f"user_{user_id}_collection"
    collection = chroma_client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=query_embedding,
        where={"book_id": book_id},
        n_results=5)
    documents = []
    for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        documents.append(Document(page_content=doc_content, metadata=metadata))
    print(documents)
    return documents


# Generate embeddings for text chunks extracted from a PDF
def generate_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])
    return chunks, embeddings


# Save embeddings and metadata for a user's PDF
def save_embeddings(user_id, book_id, pdf_path):
    collection_name = f"user_{user_id}_collection"
    collection = chroma_client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
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
                "page": chunk.metadata['page']
            },
            documents=chunk.page_content
        )
    print(f'Saved embeddings and metadata for user {user_id} and book {book_id}')
    return {f'Saved embeddings and metadata for user {user_id} and book {book_id}'}


def llm_pipeline_with_context(query: str, user_id: str, book_id: str):
    # Define the prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    context = get_relevant_documents(user_id, query, book_id)
    print(context)
    chain = RunnableMap({
        "context": lambda x: context,
        "question": lambda x: x["question"]
    }) | prompt | llm | output_parser

    # Invoke the chain and print the response
    response = chain.invoke({"question": query})
    print("Response : " + response)
    return response


# Use the LLM chain with context retrieved from ChromaDB
def llm_chain_with_context(query, user_id):
    collection_name = f"user_{user_id}_collection"
    vectordb = Chroma(client=chroma_client,
                      embedding_function=embedding_fn,
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})
    retriever = vectordb.as_retriever()

    # Define the prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    # Define the chain of operations
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }) | prompt | llm | output_parser

    # Invoke the chain and print the response
    response = chain.invoke({"question": query})
    print("Response : " + response)
    return response


# Use the LLM chain with context from ChromaDB and chat history from Redis
def llm_chain_with_context_and_memory(query: str, user_id: str, chat_id: str):
    collection_name = f"user_{user_id}_collection"
    vectordb = Chroma(client=chroma_client,
                      embedding_function=embedding_fn,
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})

    # Retrieve chat history from Redis
    history = redis.get_chat(user_id=user_id, chat_id=chat_id)

    print("History : " + str(history))

    # Define the prompt template with context and history
    template = """Answer the question based only on the following context and history:
    {context} {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    retriever = vectordb.as_retriever()

    # Define the chain of operations
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: history
    }) | prompt | llm | output_parser

    # Invoke the chain, save the chat
    response = chain.invoke({"question": query})
    redis.save_chat(user_id=user_id, chat_id=chat_id, query=query, response=response)
    print("Response : " + response)
    return response


if __name__ == "__main__":
    # redis.user_chats(user_id="001")
    llm_chain_with_context(query="summarise", user_id="001")
    # get_relevant_book(user_id="001", book_id="book_001")
    # delete_collections()
    # get_collections()
    # save_embeddings(user_id="001",
    #                 book_id="002",
    #                 pdf_path="data/romeo-and-juliet.pdf")
