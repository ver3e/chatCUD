import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
import json
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser



# Load API key from environment variables
api_key = st.secrets["secrets"]["api_key"]
CHAT_HISTORY_FILE = "chat_history.json"

# Title of the Streamlit app
st.title('Chat CUD')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
# Function to save chat history to a file
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(messages, file)

# Function to load chat history from a file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []

# Function to delete chat history file
def delete_chat_history_file():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)


def process_pdf():
     loader = PyPDFLoader("1st_Edition_Catalogue_22_23_v01.pdf")
     data = loader.load()
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500)
     chunks = text_splitter.split_documents(data)
     vector_db = FAISS.from_documents(
                                documents=chunks, 
                                embedding=embeddings,
                                
                            )
     vector_db.save_local("vector_db")
     #save the vector_db to a file
     
@st.cache_data
def generate_response(prompt, _vector_db):
    """Generates text based on the prompt, extracted text, and chat history."""
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    llm = GoogleGenerativeAI(temperature=0.7,
            model="gemini-pro",
            google_api_key=api_key,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant for Canadian university dubai. Your task is to  to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        _vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    #take the input from the user
    response=chain.invoke({"question": prompt})
    return response,prompt

   
   
    


def main():
    st.session_state.messages = load_chat_history()

   
        
    if st.sidebar.button("Process"):
                 with st.spinner("Processing..."):
                        process_pdf()

                        st.success("Processing done!")
                        
                        
    if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            delete_chat_history_file()
            st.experimental_rerun()
        
            
    if "messages" not in st.session_state:
        st.session_state.messages = [] #initialize chat history

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"): #the := operator is used to assign and check the value in the same line
        if not prompt.strip():  # Check for empty input
            st.warning("Please enter a prompt to continue.")
            return
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_chat_history(st.session_state.messages)
            with st.chat_message("user"):
                st.markdown(prompt)
            db = FAISS.load_local("vector_db",embeddings=embeddings,allow_dangerous_deserialization=True)
            response=generate_response(prompt, db)
            st.markdown(response[0])
            # Add bot message to chat history
            st.session_state.messages.append({"role": "bot", "content": response[0]})
            save_chat_history(st.session_state.messages)

            
        

    # Limit chat history (optional)
    if len(st.session_state.messages) > 50:
        st.session_state.messages = st.session_state.messages[-50:] 
    #delete chat history when re run
    
 
    # Keep the last 50 messages

if __name__ == "__main__":
    main()
    