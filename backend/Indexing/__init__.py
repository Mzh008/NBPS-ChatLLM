import os
import glob
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def auto_indexing_pipeline(
    data_dir: str = "../Data",
    chunk_size: int = 200,
    chunk_overlap: int = 50
):
    """
    自动遍历指定目录，判断文件类型并使用对应的 Loader 进行加载 -> 文本切分 -> 向量化 -> 建立检索服务。
    参数:
        data_dir: 存放各种文件的目录
        chunk_size, chunk_overlap: 用于文本切分时的参数
    返回:
        retriever_service: 一个可直接调用 get_relevant_documents(query) 的检索服务
    """

    from .utils import get_Retriever
    from .text_splitter.chinese_text_splitter import ChineseTextSplitter

    # 导入可能用到的各种自定义 Loader
    from .document_loaders.FilteredCSVloader import FilteredCSVLoader
    from .document_loaders.mydocloader import RapidOCRDocLoader
    from .document_loaders.myimgloader import RapidOCRLoader
    from .document_loaders.mypdfloader import RapidOCRPDFLoader
    from .document_loaders.mypptloader import RapidOCRPPTLoader

    # 根据扩展名 -> Loader 的映射，后续可自行扩展
    loader_map = {
        ".csv": FilteredCSVLoader,
        ".pdf": RapidOCRPDFLoader,
        ".ppt": RapidOCRPPTLoader,
        ".pptx": RapidOCRPPTLoader,
        ".jpg": RapidOCRLoader,
        ".png": RapidOCRLoader,
        ".doc": RapidOCRDocLoader,
        ".docx": RapidOCRDocLoader,
        # 如果有更多格式，可在此处继续添加
    }

    all_docs = []
    # 遍历文件夹下所有子目录及文件
    for ext, loader_class in loader_map.items():
        # glob.glob 可以匹配所有子目录
        files = glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True)
        for file_path in files:
            try:
                # 这里针对 CSV Loader 做一个简单示例，假设列名已知，可以自行调整
                # 非 .csv 文件无需传 columns_to_read
                if ext == ".csv":
                    loader = loader_class(file_path=file_path, columns_to_read=["列名1","列名2"])
                else:
                    # "metadata_columns"等等各 Loader 的其他可选参数，可按需自行补充
                    loader = loader_class(file_path=file_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")

    # 若无文件加载成功，可直接返回 None 或抛异常
    if not all_docs:
        print("未在指定目录下找到可加载的文件")
        return None

    # 文本切分
    splitter = ChineseTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pdf=False
    )
    new_docs = []
    for doc in all_docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            new_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    # 构建向量库(此处以 FAISS + OpenAIEmbeddings 为例)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(new_docs, embeddings)

    # 构建检索服务
    RetrieverServiceClass = get_Retriever("vectorstore")
    retriever_service = RetrieverServiceClass.from_vectorstore(
        vectorstore=vectorstore,
        top_k=5,
        score_threshold=0.1
    )

    return retriever_service

def indexing_pipeline(csv_path, columns_to_read, chunk_size=200, chunk_overlap=50):
    """
    演示：读取 CSV -> 文本切分 -> 向量化 -> 构建检索服务
    参数:
        csv_path: CSV 文件路径
        columns_to_read: 需要读取的列
        chunk_size, chunk_overlap: 切分文本时的分块大小与重叠
    返回:
        retriever_service: 一个可直接调用 get_relevant_documents(query) 的检索服务
    """
    from .document_loaders.FilteredCSVloader import FilteredCSVLoader
    from .text_splitter.chinese_text_splitter import ChineseTextSplitter
    from langchain.docstore.document import Document
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from .utils import get_Retriever

    loader = FilteredCSVLoader(
        file_path=csv_path,
        columns_to_read=columns_to_read
    )
    docs = loader.load()
    splitter = ChineseTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pdf=False
    )
    new_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            new_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(new_docs, embeddings)
    RetrieverServiceClass = get_Retriever("vectorstore")
    retriever_service = RetrieverServiceClass.from_vectorstore(
        vectorstore=vectorstore,
        top_k=5,
        score_threshold=0.1
    )

    return retriever_service
