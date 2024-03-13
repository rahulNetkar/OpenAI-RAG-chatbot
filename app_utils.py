import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from PyPDF2 import PdfReader

# import openai
from openai import OpenAI
import os

# Set api key
api_key = os.getenv("OPENAI_KEY")

# Initialize llm and embedding clients
client = OpenAI(api_key=api_key)
openai_ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002", api_key=api_key
)
chroma_client = chromadb.PersistentClient(path="db")

# Maintaining conversation history
conversation_history = [
    {
        "role": "system",
        "content": """You are a helpfull assistant. You are given the inputs which are
    delimited by backtiks: 
    A question from the user, some information/context
    You have to answer this question very detailed manner with respect to the given 
    information
    """,
    }
]
qr_message = [
    {
        "role": "system",
        "content": "Form a clear, concise and on point question which should contain all the important context from the following information.",
    }
]


def upload_file(file):
    loader = PdfReader(file)

    text = ""
    for page in loader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    docs = text_splitter.split_text(text=text)

    tokens = []

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=384
    )

    for text in docs:
        tokens += token_splitter.split_text(text=text)

    # creating collection
    collection = chroma_client.create_collection(
        name="text-embedding-ada-002",
        embedding_function=openai_ef,
    )

    ids = [str(i) for i in range(len(tokens))]

    collection.add(
        ids=ids,
        documents=tokens,
    )


def query_refiner(query):

    if len(conversation_history) > 1:
        conv_history = "\n".join(
            f'{entry["role"]}:{entry["content"]}' for entry in conversation_history[1:]
        )
    else:
        conv_history = " Not yet available"

    prompt = f"""
    Input query: {query} \ 
    Conversation history/log: {conv_history} . 
    """

    qr_message.append({"role": "user", "content": prompt})

    # prompt = prompt.format(query=query, history=conv_history)

    response = client.chat.completions.create(
        messages=qr_message,
        model="gpt-3.5-turbo-0125",
        temperature=0.5,
    )

    qr_message.pop()
    return response.choices[0].message.content


def update_history(role: str, content: str):
    if len(conversation_history) <= 8:
        conversation_history.append({"role": role, "content": content})
    else:
        conversation_history.pop(1)


def find_match(query):

    # role = "user"
    # content = query_refiner(query)
    # update_history(role, content)

    collection = chroma_client.get_collection(
        name="text-embedding-ada-002",
        embedding_function=openai_ef,
    )

    docs = collection.query(query_texts=query_refiner(query), n_results=5)

    return docs["documents"][0]


def generate_response(query: str):

    retrieved_doc = find_match(query)

    # conv_history = "\n".join(
    #     f'{entry["role"]}:{entry["content"]}' for entry in conversation_history
    # )

    prompt = f"""
    Question: ```{query}```,
    Informational Context: ```{retrieved_doc}```
    """

    conversation_history.append({"role": "user", "content": prompt})

    # message.format(question=query, context=retrieved_doc, chat=conv_history)

    response = client.chat.completions.create(
        messages=conversation_history,
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
    )

    update_history("assistant", response.choices[0].message.content)

    return response.choices[0].message.content


# file_path = "./2023_Annual_Report.pdf"

# upload_file(file_path)


def main():

    while True:
        question = input("Ask a question: ")

        print(generate_response(question))


main()
