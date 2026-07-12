1. What is RAG?

Answer:
RAG (Retrieval-Augmented Generation) is an AI technique that combines document retrieval with a Large Language Model. Instead of relying only on the model's training, it first retrieves relevant information from external documents and then generates an answer based on that context.

2. Why did you use RAG?

Answer:
LLMs can hallucinate and don't know private company data. RAG lets the model answer using uploaded PDFs, making responses more accurate and based on real documents.

3. Explain your project architecture.

Answer:

User
   ↓
Streamlit UI
   ↓
PDF Upload
   ↓
PyPDFLoader
   ↓
Text Splitter
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Gemma 3 (Ollama)
   ↓
Answer + Sources
4. What is LangChain?

Answer:
LangChain is a framework for building LLM applications. It provides components like document loaders, text splitters, vector stores, retrievers, prompts, and model integrations.

5. Why did you use ChromaDB?

Answer:
ChromaDB stores vector embeddings of document chunks and performs semantic similarity search to retrieve the most relevant content for a query.

6. What are embeddings?

Answer:
Embeddings are numerical vector representations of text that capture semantic meaning. Similar texts have vectors that are close in vector space.

7. Which embedding model did you use?

Answer:

sentence-transformers/all-MiniLM-L6-v2

It is lightweight, fast, and produces high-quality sentence embeddings.

8. Why did you split the PDFs?

Answer:
LLMs have context length limits. Splitting documents into chunks improves retrieval accuracy and ensures only relevant text is sent to the model.

9. What chunk size did you use?

Answer:

Chunk Size = 1000
Chunk Overlap = 200

The overlap preserves context across adjacent chunks.

10. What is chunk overlap?

Answer:
Chunk overlap repeats some text between consecutive chunks so important information isn't lost at chunk boundaries.

11. What is semantic search?

Answer:
Semantic search retrieves documents based on meaning rather than exact keyword matches by comparing vector embeddings.

12. What happens when the user asks a question?

Answer:

Convert the question to an embedding.
Search ChromaDB for similar chunks.
Retrieve the top chunks.
Build a prompt with the retrieved context.
Send it to Gemma 3.
Display the answer and sources.
13. Why did you use Ollama?

Answer:
Ollama allows me to run open-source LLMs like Gemma locally, so no API key or internet connection is required.

14. Which model did you use?

Answer:

Gemma 3 4B

It offers a good balance between performance and resource requirements.

15. Why not send the entire PDF to the model?

Answer:
Large PDFs exceed the model's context window, increase latency, and add irrelevant information. Retrieval sends only the most relevant chunks.

16. What is a vector database?

Answer:
A vector database stores embeddings and enables efficient similarity searches using vector distance metrics like cosine similarity.

17. What is the retriever?

Answer:
The retriever searches the vector database and returns the most relevant document chunks for the user's query.

18. How did you reduce hallucinations?

Answer:
I instructed the model to answer only from the retrieved context. If the answer isn't in the documents, it replies:

"I couldn't find this information in the uploaded documents."

19. What challenges did you face?

Answer:
I encountered dependency issues, ChromaDB setup problems, and Ollama CUDA crashes. I resolved the package issues and added error handling for the application. The remaining CUDA instability was related to the local Ollama runtime rather than the RAG pipeline.

20. If you had more time, what would you improve?

Answer:

Conversation memory
Streaming responses
Hybrid search (keyword + semantic)
User authentication
Docker deployment
Cloud deployment
Better source highlighting
Support for more document formats (Word, PPT, TXT)