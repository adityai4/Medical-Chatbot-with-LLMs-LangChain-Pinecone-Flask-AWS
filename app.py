from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_google_genai import ChatGoogleGenerativeAI
import os

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Load embeddings and Pinecone index
embeddings = download_embeddings()
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)
retriever = docsearch.as_retriever()

# Use a valid Gemini model
selected_model = "gemini-2.5-flash"
print(f"Using Google Gemini model: {selected_model}")

# Create the chat model
chatModel = ChatGoogleGenerativeAI(
    model=selected_model,
    temperature=0.2,
    max_output_tokens=1024
)

# Set up prompt and RAG chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get_answer", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    print(f"Received input: {msg}")
    try:
        # Handle greetings separately
        if msg in ["hi", "hello", "hey", "good morning", "good evening"]:
            return "Hello! I'm your medical assistant. How can I help you today?"
        response = rag_chain.invoke({"input": msg})
        print("RAG chain response:", response)
        answer = response.get("answer", "No answer returned.")
        print("Final answer:", answer)
        return str(answer)
    except Exception as e:
        print("Error in RAG chain:", e)
        return "Sorry, there was an error processing your request."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
