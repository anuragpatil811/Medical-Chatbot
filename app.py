import streamlit as st
import os
from medical_chatbot import MedicalChatbot
# python -m streamlit run app.py
# Set page config
st.set_page_config(
    page_title="Medical Encyclopedia Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def initialize_chatbot():
    """Initialize the medical chatbot"""
    try:
        with st.spinner("Initializing medical chatbot... This may take a few minutes."):
            st.session_state.chatbot = MedicalChatbot()
            st.session_state.initialized = True
            st.success("Medical chatbot initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.session_state.initialized = False

def main():
    # Title and header
    st.title("🏥 Medical Encyclopedia Chatbot")
    st.markdown("### Powered by Gale Encyclopedia of Medicine & Mistral-7B-Instruct")
    
    # Medical disclaimer
    with st.expander("⚠️ Important Medical Disclaimer", expanded=False):
        st.warning("""
        **IMPORTANT MEDICAL DISCLAIMER:**
        
        This chatbot is for educational and informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. 
        
        - Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        - Never disregard professional medical advice or delay in seeking it because of something you have read here.
        - If you think you may have a medical emergency, call your doctor or emergency services immediately.
        
        The information provided is based on the Gale Encyclopedia of Medicine and may not reflect the most current medical developments.
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This chatbot uses:
        - **Gale Encyclopedia of Medicine** as knowledge base
        - **Mistral-7B-Instruct** for natural language understanding
        - **LangChain** for retrieval-augmented generation
        - **Vector embeddings** for efficient information retrieval
        """)
        
        st.header("How to use")
        st.markdown("""
        1. Wait for the chatbot to initialize
        2. Ask any medical question
        3. The chatbot will search the encyclopedia and provide relevant information
        4. Each response includes source references
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chatbot if not done
    if not st.session_state.initialized:
        if st.button("Initialize Medical Chatbot"):
            initialize_chatbot()
        return
    
    # Chat interface
    st.subheader("💬 Ask your medical question")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sources" in message:
                st.markdown(message["content"])
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:** {source}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical encyclopedia..."):
                try:
                    response = st.session_state.chatbot.get_response(prompt)
                    
                    # Display response
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:** {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()