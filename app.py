import streamlit as st
import os
from rag import search_duckduckgo, rag_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Anime_BOT",layout="centered" )
st.title("Anime Chatbot")
st.caption("Enter a URL , ask any question")
st.image("raiden-shogun-neon-5120x2880-20921.jpg",width=1000)
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""
user_url = st.text_input("Enter Anime URL",key="url_input")
if user_url:
    if user_url != st.session_state.processed_url:
        with st.spinner("Processing URL..."):
            retriever = rag_chain(user_url)
            if retriever:
               st.session_state.retriever = retriever
               st.session_state.processed_url = user_url
               st.success("URL processed successfully!")
            else:
                st.error("Failed to process the URL. Please check the URL and try again.")
                st.session_state.processed_url = ""
    if st.session_state.retriever and st.session_state.processed_url == user_url:
        st.markdown('-----')
        question = st.text_input("Ask your question about the Anime here:", key=f"query_(st.session_state.processed_url)")
        if question:
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = st.session_state.retriever.invoke(question)
                    context = "\n".join([doc.page_content for doc in retrieved_docs])

                    duckduckgo_res = search_duckduckgo(question)

                    combined_context = f"RAG : {context}\n\nWeb :{duckduckgo_res}"
                    llm = ChatGroq(model ="llama-3.3-70b-versatile", temperature=1, api_key=st.secrets["API"])
                    prompt = ChatPromptTemplate.from_template(
                        """
                        Answer the following question based on the provided context.
                        Use both retrieved documents and web search results.
                        Answer in Bullet points only not in paragraph.
                        Always tell source of your answer between [Search , RAG].
                        If the information isn't in the context, say you couldn't find it.

                        Context:
                        {context}

                        Question: {input}

                        Answer:
                        """
                    )
                    fomatted_prompt = prompt.format_prompt(input=question, context=combined_context)
                    response = llm.invoke(fomatted_prompt.to_messages())
                    st.markdown('***Answer***')
                    st.success(response.content)
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
elif not user_url and st.session_state.processed_url:
    st.info("Enter an Anime URL to continue.")
    st.session_state.retriever = None
    st.session_state.processed_url = ""
else:
    st.info("Enter an Anime URL to continue.")
st.markdown('-----')
st.caption("Developed by 5132K")
