import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class RetrievalTool:
    def __init__(self, index_path: str):
        # 验证索引路径是否存在
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"未找到索引文件：{index_path}")

        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        try:
            # 加载索引
            result = FAISS.load_local(
                os.path.dirname(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 分析返回结果
            print("FAISS.load_local 返回值类型:", type(result))
            if isinstance(result, tuple):
                print("返回值为元组，长度:", len(result))
                for i, item in enumerate(result):
                    print(f"第 {i + 1} 项：{repr(item)}")
                self.vector_store = result[0]
            else:
                self.vector_store = result

        except Exception as e:
            print(f"加载索引失败: {e}")
            raise

        # 初始化检索器
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

    def run(self, query: str) -> str:
        try:
            docs = self.retriever.get_relevant_documents(query)
            results = []
            for i, d in enumerate(docs):
                snippet = d.page_content[:400].replace("\n", " ")
                results.append(f"Doc{i + 1}: source={d.metadata.get('source', '')} | content={snippet} ...")
            return "\n".join(results)
        except Exception as e:
            print(f"文档检索失败: {e}")
            return "检索失败"


if __name__ == "__main__":
    index_path = "../Database/"  # 替换为您的实际路径
    try:
        tool = RetrievalTool(index_path)
        print("初始化成功")
    except Exception as e:
        print("初始化失败:", e)
