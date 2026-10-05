import os
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Telecom Support Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Endpoint URL
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #0e76a8;
        padding: 10px;
        margin-top: 5px;
        border-radius: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/antenna.png", width=64)
st.sidebar.title("Configuration")
st.sidebar.markdown(
    "Adjust RAG retrieval parameters for response generation."
)

top_k = st.sidebar.slider(
    "Top-K Chunks to Retrieve",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of relevant policy documents pulled from Pinecone.",
)

if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Header
st.title("📡 Telecom Support Intelligence")
st.caption("AI-Powered RAG Engine for Tier-1 Customer Support & Policy Guidance")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Telecom Support Assistant. How can I help you with account management, eSIM activation, roaming policies, or device troubleshooting today?",
            "sources": [],
        }
    ]

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Display source citations if available
        if message.get("sources"):
            with st.expander("📚 Retrieved Knowledge Sources"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**Source {idx}:** `{src.get('policy_id', 'N/A')}` | **Relevance Score:** `{src.get('score', 0.0):.4f}`"
                    )
                    st.caption(f"_{src.get('text', '')}_")

# Process User Query
if prompt := st.chat_input("Ask about policies, roaming, eSIMs, or billing..."):
    # Display user query in chat
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving verified policy documents & generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt, "top_k": top_k},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response generated.")
                    sources = data.get("sources", [])

                    st.write(answer)

                    if sources:
                        with st.expander("📚 Retrieved Knowledge Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(
                                    f"**Source {idx}:** `{src.get('policy_id', 'N/A')}` | **Relevance Score:** `{src.get('score', 0.0):.4f}`"
                                )
                                st.caption(f"_{src.get('text', '')}_")

                    # Save to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                elif response.status_code == 422:
                    st.error("Validation Error: Please check your query input.")
                else:
                    st.error(f"Backend API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Unable to connect to the FastAPI backend service. Ensure Docker container or `uvicorn main:app` is running on port 8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")