import re
import os
from typing import List, Dict, Any
from langchain.schema import Document

def extract_document_ids(text: str) -> List[int]:
    """
    从文本中提取文档ID引用
    Extract document IDs from text references
    
    :param text: 包含文档引用的文本
    :return: 提取的文档ID列表
    """
    pattern = r'Doc(\d+)'
    matches = re.findall(pattern, text)
    
    if matches:
        doc_ids = [int(match) for match in matches]
        # 移除重复项但保持顺序
        unique_ids = []
        for doc_id in doc_ids:
            if doc_id not in unique_ids:
                unique_ids.append(doc_id)
        return unique_ids
    
    return []

def get_source_documents(text: str, doc_list: List[Document]) -> List[Dict[str, Any]]:
    """
    根据文本中引用的文档ID提取源文档信息
    Extract source documents based on document IDs mentioned in the text
    
    :param text: 包含文档引用的文本
    :param doc_list: 可用的Document对象列表
    :return: 源文档信息列表
    """
    doc_ids = extract_document_ids(text)
    sources = []
    
    for doc_id in doc_ids:
        if 1 <= doc_id <= len(doc_list):
            # 从文档中提取元数据（0索引列表，1索引引用）
            doc = doc_list[doc_id - 1]
            source_path = doc.metadata.get('source', 'Unknown')
            
            # 从路径中获取文件名
            filename = os.path.basename(source_path)
            
            # 创建源信息字典
            source_info = {
                'filename': filename,
                'path': source_path,
                'page': doc.metadata.get('page', None),
                'title': doc.metadata.get('title', None),
                'date': doc.metadata.get('date', None),
                'author': doc.metadata.get('author', None)
            }
            
            # 如果尚未添加，则添加到sources
            if not any(s['path'] == source_path for s in sources):
                sources.append(source_info)
    
    return sources

def format_citations(sources: List[Dict[str, Any]]) -> str:
    """
    将源文档信息格式化为引用文本
    Format source documents into citation text
    
    :param sources: 源文档信息列表
    :return: 格式化的引用文本
    """
    if not sources:
        return "Cant generate。"
    
    citations = ["Source"]
    
    for i, source in enumerate(sources):
        # 基本引用，包括编号和文件名
        citation = f"[{i+1}] {source['filename']}"
        
        # 如果有页码，添加页码信息
        if source['page']:
            citation += f", {source['page']}Page"
            
        # 如果有作者，添加作者信息
        if source['author']:
            citation += f", {source['author']}"
            
        # 如果有日期，添加日期信息
        if source['date']:
            citation += f" ({source['date']})"
            
        citations.append(citation)
    
    return "\n".join(citations)
