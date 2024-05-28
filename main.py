import json
import os
import chromadb
import redis
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI

load_dotenv('.env')

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

redis_client = redis.Redis(host="localhost", port=6379, db=0)

chroma_client = chromadb.HttpClient(host="localhost", port=3000)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embedding_fn = OpenAIEmbeddings(model='text-embedding-3-large')

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def get_relevant_documents(query, user_id, book_id):
    query_embedding = embedding_fn.embed_query(query)
    collection_name = f"user_{user_id}_collection"
    collection = chroma_client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=query_embedding,
        where={"book_id": book_id},
        n_results=5)
    # documents = []
    # for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
    #     documents.append(Document(page_content=doc_content, metadata=metadata))
    documents = []
    for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        document = {
            "page_content": doc_content,
            "metadata": metadata
        }
        documents.append(document)
    print(documents)
    return documents


# Create a collection for a user in ChromaDB
def create_collection(user_id):
    collection_name = f"user_{user_id}_collection"
    chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return collection_name


def get_collections():
    try:
        collections = chroma_client.list_collections()
        print(collections)
        return collections
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_collection(user_id):
    try:
        collection_name = f"user_{user_id}_collection"
        chroma_client.delete_collection(collection_name)
        print("Collection deleted successfully.")
        return {f"{collection_name} deleted successfully."}
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_collections():
    try:
        collections = get_collections()
        for collection in collections:
            print(collection.name)
            chroma_client.delete_collection(collection.name)
        print("Collections deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = " ".join(doc.page_content for doc in documents)
    return text


def summarize_text(text):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Respond with a modification of the story based on the user's prompt, altering the context, characters, or storyline as necessary."
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{text} and Identify the main characters"
            }
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1
    )

    summary = response.choices[0].message.content
    print('Summary : ' + summary)
    return summary


def extract_main_characters(text):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Respond to the queries of the story based on the user's prompt."
            },
            {
                "role": "user",
                "content": f"Identify the main characters in the following text:\n\n{text}"
            }
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1
    )
    characters = response.choices[0].message.content
    print('Characters : ' + characters)
    return characters.split(', ')


def chat_with_embeddings(text):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Respond with a modification of the story based on the user's prompt, altering the context, characters, or storyline as necessary."
            },
            {
                "role": "user",
                "content": f"{text}." + "Change the characters name Romeo and Juliet to ramanan, chandrika and change the context like this story happened in kerala and both are from fisherman family"
            }
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1
    )

    chat_response = response.choices[0].message.content
    print("Response Chat : " + chat_response)
    return chat_response


# Generate embeddings for text chunks extracted from a PDF
def generate_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])
    return chunks, embeddings


# Save the embeddings and metadata to a user's collection in ChromaDB
def save_embeddings(user_id, book_id, pdf_path):
    collection_name = f"user_{user_id}_collection"
    collection = chroma_client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    chunks, embeddings = generate_embeddings(pdf_path)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        document_id = f"{book_id}_{i}"
        collection.add(ids=document_id,
                       embeddings=embedding,
                       metadatas={
                           "book_id": book_id,
                           "user_id": user_id,
                           "source": chunk.metadata['source'],
                           "page": chunk.metadata['page']},
                       documents=chunk.page_content)
    print(f'Saved embeddings and metadata for user {user_id} and book {book_id}')
    return {f'Saved embeddings and metadata for user {user_id} and book {book_id}'}


def llm_chain_with_context(query, user_id):
    collection_name = f"user_{user_id}_collection"
    vectordb = Chroma(client=chroma_client,
                      embedding_function=embedding_fn,
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})
    retriever = vectordb.as_retriever()
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
    print(response)


def save_chat_history(user_id, chat_id, query, response):
    chat_key = f"user:{user_id}:chats"
    if not redis_client.exists(chat_key):
        chat_history = [{"query": query, "response": response}]
        redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
    else:
        chat_data = redis_client.hget(chat_key, chat_id)
        if chat_data:
            chat_history = json.loads(chat_data)
            if isinstance(chat_history, list):
                chat_history.append({"query": query, "response": response})
                redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
            else:
                chat_history = [{"query": query, "response": response}]
                redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
        else:
            chat_history = [{"query": query, "response": response}]
            redis_client.hset(chat_key, chat_id, json.dumps(chat_history))


def get_chat_history_for_user(user_id):
    chat_key = f"user:{user_id}:chats"
    if redis_client.exists(chat_key):
        chat_history = redis_client.hgetall(chat_key)
        chat_history = {chat_id.decode("utf-8"): json.loads(chat_data) for chat_id, chat_data in chat_history.items()}
        return chat_history
    else:
        return {}


def get_specific_chat(user_id, chat_id):
    chat_key = f"user:{user_id}:chats"
    if redis_client.exists(chat_key):
        chat_data = redis_client.hget(chat_key, chat_id)
        if chat_data:
            return json.loads(chat_data.decode("utf-8"))
    return None


def llm_chain_with_context_and_memory(query, user_id, chat_id):
    collection_name = f"user_{user_id}_collection"
    vectordb = Chroma(client=chroma_client,
                      embedding_function=embedding_fn,
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})

    history = get_specific_chat(user_id=user_id, chat_id=chat_id)
    print("History : \n" + str(history))

    template = """Answer the question based only on the following context and history:
    {context} {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    output_parser = StrOutputParser()
    retriever = vectordb.as_retriever()
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: history
    }) | prompt | llm | output_parser
    response = chain.invoke({"question": query})
    save_chat_history(user_id=user_id, chat_id=chat_id, query=query, response=response)
    print(response)
    return response


if __name__ == "__main__":
    get_chat_history_for_user(user_id="001")
    # llm_chain_with_context_and_memory(query="explain the second one mentioned above?", user_id="001", chat_id="001")
    # get_relevant_documents(user_id="001", book_id="book_001", query="what are the different topics covered ?")
    # delete_collections()
    # get_collections()
    # save_embeddings(user_id="001",
    #                 book_id="book_001",
    #                 pdf_path="data/wisdom.pdf")
