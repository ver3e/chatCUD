import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import json



# Load API key from environment variables
api_key = st.secrets["secrets"]["api_key"]
CHAT_HISTORY_FILE = "chat_history.json"

# Title of the Streamlit app
st.title('Oryx AI')

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
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=api_key)
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_documents(pdf_file):
    """Loads documents from a PDF file."""
    loader = PyPDFLoader(pdf_file.name)
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500, length_function=len, add_start_index=True)
    docs=text_splitter.split_documents(documents)
    
    return docs

@st.cache_data
def generate_response(prompt, text, chat_history):
    """Generates text based on the prompt, extracted text, and chat history."""
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

    combined_text = ""
    
    # Add chat history to the context
    for message in chat_history:
        combined_text += f"{message['content']}\n"
    
    combined_text += (
        "As your helpful assistant, I want to ensure our communication is clear and efficient. Here are some guidelines to follow:\n\n"
        "- Please provide thorough and structured answers, avoiding shortcuts and ensuring accuracy.\n"
        "- Format your responses in markdown for better readability.\n"
        "- When explaining steps or processes, provide detailed instructions.\n"
        "- Always clearly indicate the user prompt for each question.\n"
        "- Generate responses without including labels such as 'User Prompt:' and 'Bot Response:'.\n"
        "- Avoid duplicating the user prompt in your response.\n"
        "sometime you will be asked to response to prompts that could be not a question or order just response to it.\n\n"
        "- When dealing with mathematical content, use TeX functions supported by KaTeX to represent equations. Wrap LaTeX expressions with $$ and present them step by step.\n"
        "- Ensure your responses are relevant to the given prompt.\n\n"
        "Please keep these guidelines in mind when generating your responses.and make sure to provide steps if you used any. \n\n"
        "Your task prompt is:\n\n"
        f"{prompt}\n"
    )

    if text.strip():
        combined_text += f"```{text}```"

    generated_text = ""
    for chunk in llm.stream(combined_text):
        generated_text += chunk
    return generated_text, prompt


def main():
    delete_chat_history_file()
    st.session_state.messages = load_chat_history()

    with st.sidebar:
        pdf_file = st.file_uploader("Choose a PDF file (optional)", type="pdf")
        if pdf_file:
            with open(pdf_file.name, mode='wb') as w:
                w.write(pdf_file.getvalue())
            
            if st.sidebar.button("Process") and pdf_file is not None:
             with st.spinner("Processing..."):
                    texts = load_documents(pdf_file)
                    get_vector_store(texts)
                    st.success("Success")
        
            
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
        if pdf_file:
            
            

            new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
            docs=new_db.similarity_search(prompt)
            context_text = "\n\n--\n\n".join([doc.page_content for doc in docs])
            response = generate_response(prompt, context_text, st.session_state.messages)
            st.markdown(response[0])
            # Add bot message to chat history
            st.session_state.messages.append({"role": "bot", "content": response[0]})
            save_chat_history(st.session_state.messages) 
                
        else:
            response = generate_response(prompt, "", st.session_state.messages)
            st.markdown(response[0])
            # Add bot message to chat history
            st.session_state.messages.append({"role": "bot", "content": response[0]})
            save_chat_history(st.session_state.messages)

    # Limit chat history (optional)
    if len(st.session_state.messages) > 50:
        st.session_state.messages = st.session_state.messages[-50:]  # Keep the last 50 messages

if __name__ == "__main__":
    main()
    
