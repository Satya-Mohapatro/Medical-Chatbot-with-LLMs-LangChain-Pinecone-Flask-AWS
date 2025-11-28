from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\nRelevant medical context:\n{context}\n\nPrevious conversation:\n{chat_history}"),
        ("human", "{input}"),
    ]
)


def rag_with_memory(user_input):
    # Get relevant documents
    docs = retriever.get_relevant_documents(user_input)

    # Get chat history from memory
    history = memory.load_memory_variables({})["chat_history"]

    # Combine everything into a single prompt
    final_prompt = prompt.format_messages(
        input=user_input,
        chat_history=history,
        context=docs
    )

    # Generate LLM response
    llm_response = llm.invoke(final_prompt)

    # Save conversation to memory
    memory.save_context(
        {"input": user_input},
        {"output": llm_response.content}
    )

    return llm_response.content


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    answer = rag_with_memory(msg)
    return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)