import streamlit as st
from rag_utils import retrieval_and_generation

# Page config
st.set_page_config(
    page_title="📚 PDF Knowledge Q&A",
    page_icon="📖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("📌 Quick Guide")
    st.markdown("""
    - Place your PDFs in the `docs/` folder.  
    - Run **run_once.py** to prepare your knowledge base.  
    - Come back here and ask your questions below ⬇️.  
    """)
    st.success("✨ Tip: Try asking topic-specific questions.")

# Main title
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>📚 Ask Your PDFs</h1>", unsafe_allow_html=True)
st.write("<h5 style='text-align:center; color:#566573;'>Your personal document-based Q&A assistant</h5>", unsafe_allow_html=True)

# Input box
st.markdown("---")
query = st.text_input("💬 What would you like to know?", placeholder="e.g., Explain reinforcement learning in simple words")

# Button + Answer area
if st.button("🔍 Find Answer"):
    if query.strip() != "":
        with st.spinner("⏳ Searching your documents..."):
            answer = retrieval_and_generation(query)

        st.markdown("### 📢 Response:")
        st.info(answer)
    else:
        st.error("⚠ Please type a question before continuing.")
