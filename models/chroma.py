# models/chroma.py

# https://www.trychroma.com
# Documentation https://docs.trychroma.com/docs/overview/introduction
# Documentation https://docs.trychroma.com/docs/querying-collections/query-and-get

# --- INITIALIZE CHROMA VECTOR STORAGE FOR RETRIEVING DOCUMENTS
# Retrieve `nutritional` database created from Google Colab
collection_name = "nutritional"
retriever = get_retriever("nutritional")



# Initializes vector retriever and gets vectorized data stored in Chroma
# The `persist directory` is from the root repository path, not the Google Colab path.
def get_retriever(coll_name: str):
    vector_store = Chroma(
        collection_name=coll_name,
        embedding_function=embedding_model,
        persist_directory=f"{coll_name}_db"
    )

    # Create a retriever from the vector store
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_RESULT_CNT}
    )