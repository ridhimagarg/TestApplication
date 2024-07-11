import streamlit as st
import pickle
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
import tempfile
import os

def load_vectorstore(file):
    with file:
        return pickle.load(file)

class HybridSearch:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.documents = list(vectorstore.docstore._dict.values())
       
        # Create BM25 index
        tokenized_corpus = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, k=5):
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)  # Get more results for reranking
       
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
       
        # Combine scores
        combined_results = []
        for doc, vector_score in vector_results:
            doc_index = self.documents.index(doc)
            bm25_score = bm25_scores[doc_index]
           
            # Normalize scores (simple min-max normalization)
            normalized_vector_score = 1 - (vector_score / max(r[1] for r in vector_results))
            normalized_bm25_score = bm25_score / max(bm25_scores)
           
            # Combine scores (you can adjust weights here)
            combined_score = 0.5 * normalized_vector_score + 0.5 * normalized_bm25_score
            combined_results.append((doc, combined_score))
       
        # Sort and return top k results
        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]

def main():
    st.title("XML Content Retriever")
   
    # File uploader
    uploaded_file = st.file_uploader("Upload your vectorstore file", type="pkl")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading vector store..."):
                vectorstore = load_vectorstore(uploaded_file)
            st.success("Vector store loaded successfully!")
           
            # Create HybridSearch object
            hybrid_search = HybridSearch(vectorstore)
            st.success("Hybrid search initialized successfully!")
            
            # Search method selection
            search_method = st.radio("Select search method:", ("Vector Search Only", "Hybrid Search (Vector + BM25)"))
            
            # Query input
            query = st.text_input("Enter your query:")
            if query:
                with st.spinner("Searching..."):
                    if search_method == "Vector Search Only":
                        results = vectorstore.similarity_search_with_score(query, k=4)
                    else:  # Hybrid Search
                        results = hybrid_search.search(query, k=4)
               
                st.subheader(f"Top 4 Results using {search_method}:")
                for i, (doc, score) in enumerate(results, 1):
                    st.write(f"Result {i}:")
                    st.write(f"   Score: {score:.4f}")
                    st.write(f"   Content:  {doc.page_content}...")  # Display first 200 characters
                    st.write(f"   Source: {doc.metadata['source']}")
                    st.write("---")
        except Exception as e:
            st.error(f"Error loading the vector store: {str(e)}")
    else:
        st.info("Please upload a vectorstore file to proceed.")

if __name__ == '__main__':
    main()