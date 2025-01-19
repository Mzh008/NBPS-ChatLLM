#Indexing/embedding_file.py

import logging
import os
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# 初始化日志记录器
# Initialize the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 初始化嵌入模型
# Initialize the embedding model
logging.info("Loading embedding model...")
embedding_model = SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')  # 高精度嵌入模型 / High-accuracy embedding model


def store_embeddings_in_langchain_faiss(documents: List[Document], index_save_path: str) -> None:
    """
    为提供的文档生成嵌入，将其存储为LangChain格式的FAISS索引，并自动保存元数据。
    Generate embeddings for the given documents, store them as a LangChain-compatible FAISS index, and automatically save metadata.

    :param documents: 文档片段列表（`langchain.schema.Document` 对象）
                       A list of document chunks (`langchain.schema.Document` objects)
    :param index_save_path: FAISS 索引保存文件夹路径（包括元数据）。
                             Path to save the FAISS index folder (including metadata).
    """
    logging.info("Generating embeddings for all document chunks...")

    # 生成嵌入
    # Generate embeddings
    embeddings = embedding_model.encode(
        [doc.page_content for doc in documents],
        show_progress_bar=True,
        batch_size=32,
        device='cuda' if embedding_model.device.type == 'cuda' else 'cpu'
    )
    logging.info(f"Generated embeddings for {len(documents)} documents.")

    # 使用 LangChain 提供的 FAISS 封装类
    # Use the FAISS wrapper class provided by LangChain
    logging.info("Converting embeddings and metadata into FAISS format...")
    faiss_store = FAISS.from_documents(documents,
                                       HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    # 保存到指定路径
    # Save to the specified path
    os.makedirs(index_save_path, exist_ok=True)  # 确保路径存在 / Ensure the path exists
    faiss_store.save_local(index_save_path)
    logging.info(f"LangChain FAISS index successfully saved to '{index_save_path}'.")


if __name__ == "__main__":
    from Indexing.load_file import traverse_directory
    from split_file import auto_select_split_method, process_documents_with_splitting

    # 加载文档，按自定义规则进行分割
    # Load documents and split them using custom rules
    results = traverse_directory("../Data")
    split_method = auto_select_split_method(results, length_threshold=2000, structure_threshold=0.5)
    split_results = process_documents_with_splitting(results, split_method=split_method, chunk_size=512,
                                                     chunk_overlap=50)

    # 将文档存储为LangChain封装的FAISS向量数据库
    # Store documents as a LangChain-compatible FAISS vector database
    index_path = "../Database"  # 指定索引存储文件夹路径 / Specify the folder path to store the index
    store_embeddings_in_langchain_faiss(split_results, index_save_path=index_path)
