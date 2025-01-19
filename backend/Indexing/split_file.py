# Indexing/split_file.py

import logging
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import re

# 初始化日志
# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 初始化 HuggingFace 模型
# Initialize HuggingFace model
logging.info("Loading HuggingFace summarization model for semantic splitting...")
semantic_splitter = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")


def split_documents_by_length(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[
    Document]:
    """
    基于长度分割文档（改进：自定义分隔符/分割策略 + 仍采用RecursiveCharacterTextSplitter）
    Split documents by length (enhancement: custom separators/splitting strategy + leveraging RecursiveCharacterTextSplitter)

    :param documents: 原始加载的文档列表 / List of original loaded documents
    :param chunk_size: 每个块的最大字符数 / Maximum number of characters per chunk
    :param chunk_overlap: 每个块的重叠字符数 / Number of overlapping characters per chunk
    :return: 分割后的文档列表 / List of split documents
    """
    logging.info(f"Splitting documents by length with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # 改进点：使用自定义的分隔符列表，让分割更贴近自然段/句子的边界
    # Enhancement: Use a custom set of separators to make splitting align more naturally with paragraph/sentence edges
    # 比如我们先考虑 \n\n、\n、句号、问号、感叹号等常见分割标记
    # For instance, consider common delimiters like \n\n, \n, period, question mark, exclamation mark
    # 之后在每一级做递归拆分，保证最终块不超过 chunk_size。
    # Perform recursive splitting for each level to ensure the final chunk size does not exceed the chunk_size
    # 也可以根据具体文档类型再定制其他分隔符。
    # More separators can be defined based on specific document types
    custom_separators = ["\n\n", "\n", ".", "?", "!", "。", "？", "！"]

    # 通过自定义的 separators 初始化 RecursiveCharacterTextSplitter
    # Initialize RecursiveCharacterTextSplitter with custom separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=custom_separators
    )

    split_docs = []
    for doc in documents:
        # 对 doc 的 page_content 做切分
        # Split the page_content of the document
        chunks = text_splitter.split_text(doc.page_content)
        # 将切分后的片段重新封装成 Document 并收集
        # Rewrap split chunks as Document objects and collect
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    logging.info(f"Total chunks created: {len(split_docs)} (length-based)")
    return split_docs


def split_documents_by_semantics(documents: List[Document], max_length: int = 512) -> List[Document]:
    """
    基于 HuggingFace 模型的语义分割文档（改进：先做粗粒度段落/句子拆分，再对过长内容调用摘要模型）
    Split documents semantically based on the HuggingFace model (enhancement: coarse-grained splitting into paragraphs/sentences first, followed by summarization for long content)

    :param documents: 原始加载的文档列表 / List of original loaded documents
    :param max_length: HuggingFace 模型处理文本的最大长度 / Maximum length for text processing by HuggingFace model
    :return: 分割后的文档列表 / List of split documents
    """
    logging.info("Splitting documents by semantics using a hybrid approach (paragraph/sentence + summarization)")
    split_docs = []

    for doc in documents:
        content = doc.page_content
        # 如果文本长度已经低于 max_length，则直接保留，不调用摘要模型
        # Skip summarization if the content length is already below max_length
        if len(content) <= max_length:
            split_docs.append(doc)
            continue

        try:
            # ----------- 改进：先做段落级（或句子级）的粗分割 -----------
            # ----------- Enhancement: Perform coarse-grained splitting by paragraph (or sentence) first -----------
            # 这里以段落为例，按双换行 \n\n 分段；若需要更细可改成按句子拆分
            # Split into paragraphs based on double line-break (\n\n). For finer splits, switch to sentence-level splitting
            paragraphs = re.split(r'\n\s*\n+', content.strip())

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # 如果段落本身已经足够短，就直接纳入
                # If the paragraph is short enough, include it directly
                if len(paragraph) <= max_length:
                    split_docs.append(Document(page_content=paragraph, metadata=doc.metadata))
                else:
                    # 处理仍然过长的段落：进一步拆分或通过 summarization 生成若干段落
                    # Process overly long paragraphs: further splitting or summarization to generate several pieces
                    # 如果要多次切分，需要自己做 while len(paragraph)>模型输入限制的逻辑
                    # Loop for multi-part splitting if needed while len(paragraph) > model input limits
                    summary_output = semantic_splitter(
                        paragraph,
                        max_length=max_length,
                        min_length=int(max_length * 0.7),  # 给个范围，不要过分缩短 / Specify a range to avoid excessive brevity
                        do_sample=False
                    )
                    for item in summary_output:
                        # 将生成的摘要文本作为新的 Document
                        # Treat summarized text as a new Document
                        summarized_text = item["summary_text"].strip()
                        if summarized_text:
                            split_docs.append(Document(page_content=summarized_text, metadata=doc.metadata))

        except Exception as e:
            logging.error(f"Semantic split failed for document. Error: {e}")

    logging.info(f"Total semantic chunks created: {len(split_docs)} (semantic-based)")
    return split_docs

def process_documents_with_splitting(documents: List[Document], split_method: Optional[str] = "length", **kwargs) -> \
        List[Document]:
    """
    根据指定方式分割文档
    Split documents based on the specified method

    :param documents: 原始文档列表 / List of original documents
    :param split_method: 分割方式 ("length" 或 "semantics") / Splitting method ("length" or "semantics")
    :param kwargs: 其他分割参数 / Additional splitting parameters
        - 对于 "length" 模式可传: chunk_size, chunk_overlap
        - For "length" mode: chunk_size, chunk_overlap
        - 对于 "semantics" 模式可传: max_length
        - For "semantics" mode: max_length
    :return: 分割后的文档列表 / List of split documents
    """
    if split_method == "length":
        # 只传递适用于长度分割的参数
        # Pass only parameters suitable for length-based splitting
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        return split_documents_by_length(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif split_method == "semantics":
        # 适用于语义分割的参数无需传递 chunk_size 或 chunk_overlap
        # Parameters specific to semantic splitting (no need for chunk_size or chunk_overlap)
        max_length = kwargs.get("max_length", 512)
        return split_documents_by_semantics(documents, max_length=max_length)
    else:
        raise ValueError(f"Unsupported split method: {split_method}")


def compute_structure_score(document: Document) -> float:
    """
    对单个文档进行简单的“结构评分”，返回一个 0~1 之间的浮点数。
    Perform a simple "structure scoring" for a single document, returning a float between 0 and 1.
    分数越高，说明文档可能具有更清晰的层次结构（例如带有较多标题、空行、段落）。
    The higher the score, the more likely the document has a clear hierarchical structure (e.g., with more headings, blank lines, and paragraphs).

    仅作演示，真实环境中可结合更复杂的 NLP 分析。
    For demonstration purposes only. In real-world scenarios, combine with more sophisticated NLP analysis.
    """
    text = document.page_content
    if not text:
        return 0.0

    # 统计 Markdown 标题行（如 #、##、### 开头的）
    # Count Markdown-style headings (e.g., lines starting with #, ##, ###)
    md_headings = re.findall(r'(?m)^#{1,6}\s+\S.*$', text)
    md_heading_count = len(md_headings)

    # 统计空行，用于粗略估计段落数量
    # Count blank lines to roughly estimate the number of paragraphs
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    paragraph_count = len(paragraphs)

    # 文本总长度
    # Total length of text
    total_chars = len(text)

    # 如果文本极短，则直接视为结构不明显
    # Consider structure to be negligible if the text is extremely short
    if total_chars < 100:
        return 0.0

    # 假设标题行越多、段落越多，就认为结构越明显
    # Assume the more headings and paragraphs, the more structured the document
    # 这里做一个非常粗糙的打分示例
    # This is a very basic scoring example
    heading_score = md_heading_count / 10.0  # 如果有 10 个以上标题，则 heading_score >= 1 / If more than 10 headings, heading_score >= 1
    paragraph_score = paragraph_count / 10.0  # 超过 10 段落，也算结构明显 / Over 10 paragraphs, consider it well-structured

    # 还可以加入其他规则，比如正则匹配 "第X章/节" 等，但此处仅做演示
    # Additional rules can be added (e.g., matching patterns like "Chapter X"), but this is for demonstration only
    # 最终分数进行裁剪到 0~1 范围
    # Clip the final score to a range of 0~1
    structure_score = heading_score + paragraph_score
    if structure_score > 1.0:
        structure_score = 1.0
    return structure_score


def auto_select_split_method(documents: List[Document], length_threshold: int = 2000,
                             structure_threshold: float = 0.5) -> str:
    """
    根据文档内容自动选择最适合的分割方式。可综合以下因素：
    Automatically select the most suitable splitting method based on document content. The decision can consider:
    1. 文档的平均长度（length_threshold）
    1. Average length of documents (length_threshold)
    2. 文档的平均结构评分（structure_threshold）
    2. Average structure score of documents (structure_threshold)

    :param documents: 原始文档列表 / List of original documents
    :param length_threshold: 长度阈值，超过该阈值可能更适合语义分割 / Length threshold. Documents longer than this may be better suited for semantic splitting
    :param structure_threshold: 结构评分阈值，大于此阈值可能说明文档更有层次感，适合语义分割 / Structure score threshold. Higher scores suggest better hierarchical structure and suitability for semantic splitting
    :return: 推荐分割方式 ("length" 或 "semantics") / Recommended splitting method ("length" or "semantics")
    """
    # 计算平均长度
    # Calculate average length
    average_length = sum(len(doc.page_content) for doc in documents) / len(documents)
    logging.info(f"Average document length: {average_length:.2f} characters.")

    # 计算文档的结构评分并取平均
    # Compute structure scores for documents and calculate the average
    structure_scores = [compute_structure_score(doc) for doc in documents]
    avg_structure_score = sum(structure_scores) / len(structure_scores) if structure_scores else 0.0
    logging.info(f"Average structure score: {avg_structure_score:.2f} (range 0~1, higher => more structured)")

    # 改进点：多级判断
    # Enhancement: Multi-tiered decision-making
    # 如果文档极大（比如远超 length_threshold 的 2 倍），优先使用纯长度切分来避免超长处理
    # If the documents are extremely large (e.g., exceeding length_threshold by 2x), prioritize length-based splitting to avoid overly long inputs
    if average_length > length_threshold * 2:
        logging.info(
            "Documents are extremely large. Using length-based splitting to avoid huge input to summarization.")
        return "length"

    # 如果平均长度 > length_threshold 或者结构评分 >= structure_threshold，则使用语义分割
    # Use semantic splitting if the average length > length_threshold or structure_score >= structure_threshold
    if average_length > length_threshold or avg_structure_score >= structure_threshold:
        logging.info("Using semantic splitting due to large average document length or high structure score.")
        return "semantics"
    else:
        # 否则使用基于长度的分割
        # Otherwise, use length-based splitting
        logging.info("Using length-based splitting due to shorter document length or lower structure score.")
        return "length"


if __name__ == "__main__":
    from Indexing.load_file import traverse_directory

    # 假设 traverse_directory("../Data") 可以加载 ../Data 目录下的文件，返回 List[Document]
    # Assume `traverse_directory("../Data")` loads files in the ../Data directory and returns a List[Document]
    results = traverse_directory("../Data")

    logging.info("Starting document splitting process...")

    # 根据文档内容自动选择分割方式
    # Automatically select splitting method based on document content
    # length_threshold=2000 表示如果文档平均长度超过 2000 字符，就倾向使用语义分割
    # length_threshold=2000 means semantic splitting is preferred if average document length exceeds 2000 characters
    # structure_threshold=0.5 表示如果文档平均结构评分大于 0.5，就倾向使用语义分割
    # structure_threshold=0.5 means semantic splitting is preferred if average structure score exceeds 0.5
    split_method = auto_select_split_method(results, length_threshold=2000, structure_threshold=0.5)

    try:
        # 按自动选择的分割方式处理文档
        # Process documents using the automatically selected splitting method
        split_results = process_documents_with_splitting(
            results,
            split_method=split_method,
            chunk_size=512,  # 针对长度分割的参数 / Parameters for length-based splitting
            chunk_overlap=50,  # 针对长度分割的参数 / Parameters for length-based splitting
            max_length=512  # 针对语义分割的参数 / Parameters for semantic splitting
        )

        logging.info(f"Successfully split into {len(split_results)} parts.")

        # 展示分割后的部分结果（仅展示前 5 个，以免日志过长）
        # Display a subset of the split results (show only the first 5 to avoid overly long logs)
        for doc in split_results[:5]:
            print("Metadata:", doc.metadata)
            print("Content Preview:", doc.page_content[:200], "\n---")  # 显示前 200 个字符 / Show the first 200 characters

    except Exception as e:
        logging.error(f"Document splitting failed. Error: {e}")
