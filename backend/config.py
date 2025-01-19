# -*- coding: utf-8 -*-

#config
#配置（中文） 
#Configuration (English)

VECTOR_DATABASE = "Chroma"  #Chroma or Faiss
# 向量数据库可选: Chroma 或 Faiss
# The vector database can be chosen as: Chroma or Faiss

CHUNK_SIZE = 500
# 文本切分时的块大小（中文）
# Chunk size when splitting text (English)

CHUNK_OVERLAP = 100
# 文本切分时的重叠大小（中文）
# Overlap size when splitting text (English)

# 其他可能的全局参数可在此处添加
# Other potential global parameters can be added here

## -------------------------------- Agent
NER_MODEL = "dslim/bert-base-NER"