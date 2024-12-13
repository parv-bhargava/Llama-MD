import time
import streamlit as st
import os
import json
from botocore.exceptions import ClientError
from rag.vector_db import VectorDB
from rag.embedding import EmbeddingModel
from rag.utils import create_bedrock_client
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



model = AutoModelForCausalLM.from_pretrained("./models")
tokenizer = AutoTokenizer.from_pretrained("./models")
model.to(device)

def doctor_response_finetune(question, max_length=300, temperature=0.6):
    """
    Generates a doctor's response based on the patient's question.

    Parameters:
        question (str): The patient's question.
        max_length (int): The maximum token length of the response.
        temperature (float): Sampling temperature for diversity.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The doctor's response.
    """
    prompt = (
    "<System Prompt>\n"
    "You are an expert medical doctor of the patient.\n"
    "Read the patient's query and provide a clear, concise, and medically sound response.\n\n"
    "Your answer should include:\n"
    "- A diagnosis\n"
    "- A recommended treatment plan or next steps\n\n"
    "Do not repeat the patient's question. Avoid unnecessary disclaimers.\n"
    "Keep your answer focused, authoritative, and helpful.\n"
    "</System Prompt>\n\n"
    "Query: {question}\n\n"
    "Your Response:".format(question=question)
)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Remove the input text from the generated output
    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)
    response = re.sub(r'<[^>]+>', '', response)
    return response
def doctor_response_finetune_rag(query, max_length=300, temperature=0.6):
    """
    Generates a doctor's response based on the patient's question with context retrieved by RAG.

    Parameters:
        query (str): The patient's question.
        max_length (int): The maximum token length of the response.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The doctor's response.
    """

     #Retrieve relevant context using the vector DB
    vector_db = VectorDB(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_ENV"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        dimension=int(os.getenv("DIMENSION")),
        metric=os.getenv("METRIC"),
        cloud=os.getenv("PINECONE_CLOUD")
    )

    # Get the top-k relevant context documents from the vector DB
    search_results = vector_db(query, top_k=5)
    context = ""
    for result in search_results.get('matches', []):
        if 'metadata' in result and 'text' in result['metadata']:
            context += result['metadata']['text'] + "\n"

    if not context:
        context = "No relevant context found."


    prompt = f"""
    Question: {query}
    Context: {context}
    Your Response as an expert doctor specializing in gynecology, obstetrics, and pregnancy:

    """


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )


    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]


    response = tokenizer.decode(output_final, skip_special_tokens=True)
    response= re.sub(r'<[^>]+>', '', response)


    return response.split("Question:")[1].strip() if "Question:" in response else response


def handle_rag(query):
    st.success("RAG Model Selected")
    response=doctor_response_finetune_rag(query)
    return response


def handle_fine_tune(query):
    # Process user query in Fine Tune mode
    # return f"Fine Tune response for: {query}"
    st.success("FineTune Model Selected")
    response=doctor_response_finetune(query)
    return response
# Set page config first

def main():
    st.set_page_config(page_title="LLAMA, MD", page_icon="🤖", layout="wide")

    # Sidebar
    with st.sidebar:
        st.markdown("## About LLAMA, MD ")
        st.write(
            "LLAMA, MD is a Gynecology and Obstetrics chatbot that can answer questions about pregnancy, childbirth"
        )

        # Display tips
        with st.expander("Tips for using LLAMA, MD"):
            st.info("""
            - Ask questions related to pregnancy, childbirth, and gynecology.
            - Be polite and respectful.
            - Avoid sharing personal information.
            - If you encounter any issues, please report them.
            """)

        # Dropdown selector
        option = st.selectbox(
            "Choose a mode:",
            ("Select an option", "Fine Tune", "RAG"),
            index=0,
            help="Select a mode to activate the respective functionality.",
        )
        st.session_state["selected_mode"] = option  # Save the selected mode

        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # Main chat interface
    st.markdown(
        "<h1 style='text-align: center; font-family: Arial; font-size: 36px; font-weight: bold;'>LLAMA, MD - Your Virtual Assistant 🤖</h1>",
        unsafe_allow_html=True)

    # Initialize session state


    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("What would you like to know about pregnancy, childbirth, or gynecology?")

    if prompt:
        # Append the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response after user input
        with st.spinner("LLAMA, MD is thinking... 🤔"):
            start_time = time.time()

        # Pass the user input to the selected function
            if st.session_state["selected_mode"] == "RAG":
                response = handle_rag(prompt)
            elif st.session_state["selected_mode"] == "Fine Tune":
                response = handle_fine_tune(prompt)
            else:
                st.warning("Please select a mode from the sidebar.")
                response = "Please select a mode."

            response_time = time.time() - start_time

        # Append assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant's response
        st.chat_message("assistant").markdown(response)

        # Optionally show response time
        st.markdown(f"<p style='text-align: right; color: #888;'>Response time: {response_time:.2f}s</p>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()

