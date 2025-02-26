#Agent/ai_agent.py
import json
import logging
import os
from typing import List, Dict, Any, Optional

# ===== LangChain Related =====
# ===== 与 LangChain 相关 =====
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # 引入 LLMChain 类 (Importing LLMChain class)


# ===== Other Dependencies =====
# ===== 其他依赖 =====

# ===================== 1. JudgeDifficultyTool =====================
# ===================== 1. JudgeDifficultyTool 工具 =====================
class JudgeDifficultyTool:
    """
    A tool used to classify the difficulty of a question.
    一个用来分类问题难度的工具。
    """

    def __init__(self, llm: LLM):
        # Prompt template for defining classification rules and output format.
        # Prompt模板，用于定义分类规则和返回格式
        template = """You are a question difficulty classifier. Classify the user's question into the three levels of easy, medium, or hard.
        Please return in the following JSON format:
        {{
            "level": "easy/medium/hard",
            "reason": "...Reason..."
        }}
        Question: {question}
        Please return the classification result."""
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"]
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, query: str) -> dict:
        """
        Run the tool to classify the question difficulty.
        运行工具以对问题进行难度分类。

        :param query: User's input question
        :param query: 用户的问题输入

        :return: Dictionary with level and reason. {"level": "easy/medium/hard", "reason": "..."}
        :return: 包含level和reason的字典结果 {"level": "easy/medium/hard", "reason": "..."}
        """
        try:
            # Invoke LLM to execute the chain.
            # 调用 LLM 执行链条
            raw_result = self.chain.run({"question": query})
            logging.info(f"JudgeDifficultyTool Raw Response: {raw_result}")
            logging.info(f"JudgeDifficultyTool 原始返回结果: {raw_result}")

            # Try to parse the result into JSON format.
            # 尝试解析结果为 JSON
            if isinstance(raw_result, str):
                parsed_result = json.loads(raw_result)  # Convert to dict
                # 转换为字典
            else:
                raise ValueError(f"Result is not in JSON format: {raw_result}")
                # 返回值非JSON格式

            # Extract `level` and validate the value.
            # 提取 level，并验证值是否有效
            level = parsed_result.get("level", "medium").lower()
            if level not in {"easy", "medium", "hard"}:
                raise ValueError(f"Invalid level value: {level}")
                # 无效的level值

            # Return the classification result.
            # 返回分类结果
            return {
                "level": level,
                "reason": parsed_result.get("reason", "No reason provided.")
                # 未提供理由
            }

        except json.JSONDecodeError:
            # Handle failure in parsing raw_result to JSON.
            # 处理JSON解析失败的情况
            logging.error(f"Failed to parse JudgeDifficultyTool response to JSON: {raw_result}")
            logging.error(f"JudgeDifficultyTool返回值解析为JSON失败: {raw_result}")
            return {
                "level": "medium",  # Default classification
                # 默认分类
                "reason": "Failed to parse the response. Using default classification."
                # 返回值解析失败，使用默认分类。
            }

        except Exception as e:
            # Handle general exceptions.
            # 处理一般异常
            logging.error(f"Error occurred while invoking JudgeDifficultyTool: {e}")
            logging.error(f"调用JudgeDifficultyTool时发生错误: {e}")
            return {
                "level": "medium",  # Default classification
                "reason": "System encountered an error. Using default classification."
                # 系统发生错误，使用默认分类。
            }


def create_judge_difficulty_tool(llm: LLM) -> Tool:
    """
    Create and return an instance of Tool that encapsulates the functionality of JudgeDifficultyTool.
    创建并返回一个 Tool 对象的实例，封装 JudgeDifficultyTool 的功能。

    :param llm: LLM model instance
    :param llm: LLM 模型实例

    :return: LangChain Tool wrapper class
    :return: LangChain 工具包装类 Tool
    """
    jd_tool = JudgeDifficultyTool(llm)
    return Tool(
        name="JudgeDifficultyTool",
        func=jd_tool.run,
        description=(
            "Classify the difficulty of questions (Returns JSON, "
            "根据用户问题来判断难度(返回 JSON, "
            "{\"level\": \"easy/medium/hard\", \"reason\": \"...\"})."
        )
    )


# ===================== 2. RewriteTool =====================
# ===================== 2. RewriteTool 工具 =====================

class QuestionRewriteTool:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, query: str) -> str:
        # Rewrite prompt to simplify and retain the core meaning of the user's question.
        # 用于简化并保留用户问题核心含义的 Prompt。
        rewrite_prompt = f"""You are a question rewrite assistant. Simplify and clarify the following user's question while retaining its core meaning:
User's Question: {query}
Rewritten Question:"""
        # 您是一个问题重写助手，请将以下用户问题改写得简洁明了并保留核心含义。
        return self.llm(rewrite_prompt)


def create_question_rewrite_tool(llm: LLM) -> Tool:
    """
    Create and return an instance of a tool for rewriting questions.
    创建并返回一个工具实例，用于改写问题。

    :param llm: LLM model instance
    :param llm: LLM 模型实例

    :return: LangChain tool encapsulation class
    :return: LangChain 工具包装类 Tool
    """
    rewrite_tool = QuestionRewriteTool(llm)
    return Tool(
        name="RewriteTool",
        func=rewrite_tool.run,
        description="A tool for rewriting/simplifying user questions. Input is the original question, and the output is the rewritten question."
        # 对用户问题进行改写/精简的工具。输入为原始问题，输出为改写后的问题。
    )


# ===================== 3. HyDETool =====================
# ===================== 3. HyDETool 工具 =====================

class HyDETool:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, query: str) -> str:
        # HyDE prompt to write hypothetical answers for assisting retrieval.
        # HyDE 提示，用于生成假设回答以帮助检索。
        hyde_prompt = f"""You are a language model that assists with retrieval. Based on the following question, write a possible (not necessarily correct) hypothetical paragraph to aid retrieval:
Question: {query}
Hypothetical Answer (as detailed as possible):"""
        return f"【Question】{query}\n【HyDE】{self.llm(hyde_prompt)}"
        # 【问题】{query}\n【HyDE】{self.llm(hyde_prompt)}


def create_hyde_tool(llm: LLM) -> Tool:
    """
    Create and return an instance of a HyDE tool for generating hypothetical answers.
    创建并返回 HyDE 工具实例，用于生成假设回答。

    :param llm: LLM model instance
    :param llm: LLM 模型实例

    :return: HyDE tool encapsulated as a LangChain tool.
    """
    hyde_tool = HyDETool(llm)
    return Tool(
        name="HyDETool",
        func=hyde_tool.run,
        description="Generates HyDE (hypothetical answers) for a question and outputs text in the form '【Question】...\\n【HyDE】...'."
        # 为问题生成HyDE(假设答案)，输出 '【问题】...\\n【HyDE】...' 形式的文本。
    )
# ===================== 4. RetrievalTool =====================
# ===================== 4. 检索工具 RetrievalTool =====================
class RetrievalTool:
    def __init__(self, index_path: str):
        """
        Initialize the RetrievalTool.
        初始化 RetrievalTool。

        :param index_path: Path to the FAISS index file
        :param index_path: FAISS 索引文件路径

        :raises FileNotFoundError: If the specified FAISS index file is not found
        :raises FileNotFoundError: 如果未找到指定的 FAISS 索引文件
        """
        if not os.path.exists(index_path):  # 检查索引文件是否存在
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            # 未找到FAISS索引文件: {index_path}

        # Initialize embeddings using the 'sentence-transformers' pre-trained model.
        # 使用 'sentence-transformers' 预训练模型初始化嵌入
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Load the FAISS vector store from the specified path.
        # 显式允许反序列化并加载 FAISS 向量存储
        self.vector_store = FAISS.load_local(
            os.path.dirname(index_path),  # Load from the adjusted path
            # 加载利用已调整路径
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        # Set the retriever with a search limit of top 6 results.
        # 设置检索器，限制最多返回6个结果
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

    def run(self, query: str) -> str:
        """
        Run the RetrievalTool to fetch relevant documents based on the query.
        运行检索工具，根据查询获取相关文档。

        :param query: User query text
        :param query: 用户查询文本

        :return: A string of retrieved documents (with metadata and content snippets)
        :return: 检索到的文档列表字符串（带有元数据和内容片段）
        """
        docs = self.retriever.get_relevant_documents(query)  # 获取相关文档
        results = []

        # Iterate through retrieved documents and prepare a formatted list.
        # 遍历检索到的文档并准备格式化列表
        for i, d in enumerate(docs):
            snippet = d.page_content[:400].replace("\n", " ")  # 截取文档内容前400个字符并去掉换行符
            results.append(f"Doc{i + 1} | source={d.metadata.get('source', '?')} | content={snippet}...")
            # 格式化文档信息：索引、source元数据、内容片段

        return "\n".join(results)  # 拼接所有文档信息并返回


def create_retrieval_tool(index_path: str) -> Tool:
    """
    Create and return the retrieval tool for fetching documents from a vector database.
    创建并返回一个检索工具，用于从向量数据库中检索文档。

    :param index_path: Path to the FAISS index to be used for retrieval
    :param index_path: FAISS 索引的路径，用于检索

    :return: LangChain Tool object
    :return: LangChain 工具对象
    """
    rt_tool = RetrievalTool(index_path)
    return Tool(
        name="RetrievalTool",
        func=rt_tool.run,
        description=(
            "A tool for retrieving documents from the vector store. "
            "在向量库中检索文档的工具。"
            "Input is the query; the output is the list of retrieved documents with source and content snippets."
            "输入为查询，输出为检索到的文档列表（带 source 和内容片段）。"
        )
    )


# ===================== 5. ReRankingTool =====================
# ===================== 5. 重排序工具 ReRankingTool =====================
class ReRankingTool:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, input_text: str) -> str:
        # Prompt 用于根据问题相关度重新排序文档
        re_prompt = f"""
You are a document re-ranking tool. Please re-order the following documents based on relevance to the question, placing the most relevant ones at the beginning.
只输出重排后的文档列表(文本原样即可)，不做额外解释。
Only output the re-ordered document list (keep the text as it is), without additional explanation.

{input_text}

Re-ordered result:
重排结果:
"""
        return self.llm(re_prompt)


def create_reranking_tool(llm: LLM) -> Tool:
    """
    Create and return the ReRanking Tool instance that encapsulates the document reordering feature.
    创建并返回重排序工具的实例，封装文档重新排序功能。

    :param llm: LLM model instance
    :param llm: LLM 模型实例

    :return: LangChain Tool instance
    :return: LangChain 工具实例
    """
    rr_tool = ReRankingTool(llm)
    return Tool(
        name="ReRankingTool",
        func=rr_tool.run,
        description=(
            "Performs second-level sorting (Re-ranking) of retrieved documents. "
            "对检索到的文档进行二次排序(Re-ranking)。"
            "Input should include 'question' and 'document list'. Output is a re-ordered document list."
            "输入应包含'问题'和'文档列表'，输出为重排后的文档列表。"
        )
    )


# ===================== 6. AnswerTool =====================
# ===================== 6. 回答工具 AnswerTool =====================
class AnswerTool:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, input_text: str) -> str:
        # Prompt 用于根据问题和参考文档生成最终答案
        prompt = f"""
You are a professional question-answering system. Below are a user question and referential document content. 
你是一位专业问答系统。以下是用户问题和可参考的文档内容。
Based on the document content, please try your best to answer the question; if there is insufficient information, provide a well-reasoned guess and specify any uncertainties.
请你基于文档内容尽力回答问题；若信息不足，也请给出推测并明确不确定之处。

{input_text}

Please provide a concise but complete final answer:
请给出简明但完整的最终答案:
"""
        return self.llm(prompt)


def create_answer_tool(llm: LLM) -> Tool:
    """
    Create a tool for generating final answers based on user input and documents.
    创建一个工具，用于基于用户输入和文档内容生成最终回答。

    :param llm: LLM model instance
    :param llm: LLM 模型实例

    :return: LangChain Tool instance
    :return: LangChain 工具实例
    """
    at_tool = AnswerTool(llm)
    return Tool(
        name="AnswerTool",
        func=at_tool.run,
        description=(
            "Generates final answers based on user questions and document content. "
            "基于用户问题和文档内容，产生最终回答的工具。"
            "Input should include 'user question' and 'document list'. Output is the final answer."
            "输入需包含'用户问题'和'文档列表'。输出为最终答案。"
        )
    )


# ===================== LanguageDetectionTool =====================
# ===================== 语言检测工具 =====================
import re


class LanguageDetectionTool:
    """
    A tool to detect if the input is in Chinese or English.
    用于检测输入是中文还是英文的工具。
    """

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect the language type of the input text.
        :param text: User's input text
        :return: "chinese" or "english"
        """
        if re.search(r'[\u4e00-\u9fff]', text):  # Check if characters belong to Chinese
            return "chinese"
        elif re.fullmatch(r'[A-Za-z\s]+', text):  # Check if it is pure English
            return "english"
        else:
            return "mixed"  # 当内容包括中英文或无法分类时返回

    def run(self, text: str) -> str:
        """
        Use the detection feature and return a preferred response language.
        :param text: Input text for detection
        :return: Suggested response language
        """
        language = self.detect_language(text) 
        if language == "chinese":
            return "回答语言：中文" 
        elif language == "english":
            return "Response Language: English"  
        else:
            return "Unable to determine a clear language. Suggest handling both Chinese and English content separately."


def create_language_detection_tool() -> Tool:
    """
    Creates and returns a LangChain Tool encapsulating the functionality of LanguageDetectionTool.
    创建并返回一个 LangChain Tool，封装了 LanguageDetectionTool 的功能。

    :return: LangChain Tool instance
    :return: LangChain 工具实例
    """
    lang_tool = LanguageDetectionTool()
    return Tool(
        name="LanguageDetectionTool",
        func=lang_tool.run,
        description="Detect whether the input text is in Chinese or English, and returns the suggested response language."
        # 检测输入语言类型是否为中文或英文，并返回建议回答语言。
    )

# ===================== 7. 构建支持“完整思维链”风格的 RAG Agent =====================
# ===================== 7. Build an advanced "Chain of Thought" RAG Agent =====================
def build_advanced_rag_agent_chain_of_thought(index_path: str,
                                              openai_api_key: str,
                                              temperature: float = 0.1,
                                              window_size: int = 4) -> AgentExecutor:
    """
    Create a RAG Agent that produces a more detailed "Chain-of-Thought" response.
    创建一个可以在回答中输出更完整思维链的RAG Agent示例。

    (In production, the reasoning process is usually hidden from the user.
    This example explicitly demonstrates the model's reasoning for development/presentation purposes.)
    (在生产环境里, 通常不会直接向用户展示全部推理过程。
    这里为了开发/演示, 显式要求模型给出详细推理(Thought)。)
    """

    # ---------- (A) Initialize the underlying LLM ----------
    # ---------- (A) 初始化底层 LLM ----------
    base_llm = OpenAI(
        openai_api_key=openai_api_key,
        temperature=temperature
    )

    # ---------- (B) Create Tools ----------
    # ---------- (B) 构建工具 ----------
    judge_tool = create_judge_difficulty_tool(base_llm)  # 判断问题难度工具
    rewrite_tool = create_question_rewrite_tool(base_llm)  # 重写问题增强检索工具
    hyde_tool = create_hyde_tool(base_llm)  # 假设性回答增强检索工具
    retrieval_tool = create_retrieval_tool(index_path)  # 检索向量库文档工具
    rerank_tool = create_reranking_tool(base_llm)  # 重排文档顺序工具
    answer_tool = create_answer_tool(base_llm)  # 生成最终答案工具
    lang_detect_tool = create_language_detection_tool()  # 输入语言检测工具

    tools = [
        lang_detect_tool,
        judge_tool,
        rewrite_tool,
        hyde_tool,
        retrieval_tool,
        rerank_tool,
        answer_tool
    ]

    # ---------- (C) Memory ----------
    # Using window memory
    window_memory = ConversationBufferWindowMemory(
        k=window_size,  # window size
        memory_key="chat_history",  # memory key
        return_messages=True  # message returned
    )
    summary_memory = ConversationSummaryMemory(llm=base_llm, memory_key="summary")

    # ---------- (D) Prompt (Chain-of-Thought 风格) ----------
    # ---------- (D) 提示模板 (连贯推理风格) ----------
    prefix = """You are an advanced RAG Agent equipped with the following tools:
1) LanguageDetectionTool: Detect the language type and adjust response language accordingly.
2) JudgeDifficultyTool: Judge the difficulty of the question (easy/medium/hard).
3) RewriteTool: Rewrite questions to improve retrieval quality.
4) HyDETool: Generate hypothetical answers to enhance retrieval.
5) RetrievalTool: Retrieve documents from the vector database (returns document lists only).
6) ReRankingTool: Re-rank the document list.
7) AnswerTool: Generate final answers based on document content.

Your goal:
- Use the JudgeDifficultyTool to determine question difficulty.
- For more complex queries, rewrite or apply HyDE, then retrieve, re-rank, and answer the question.
- For simpler queries, directly retrieve and answer, or answer without any tools.
- Remember: Always provide detailed reasoning (Thought) and use tools (Action) as necessary.
- At the end, write your final answer (Final Answer). Detect the input language and adjust your response accordingly.

以下是当前对话的总体摘要(不一定有内容):
-----
{summary}
-----

最近几轮对话内容:
-----
{chat_history}
-----

When answering, begin with reasoning (Thought). If using tools, write:
Action: <tool_name>
Action Input: <input_for_tool>
Next, write:
Observation: <results_from_tool>
... (Repeat Action/Observation cycles as needed)
Finally, write:
Final Answer: <your_answer>
"""

    suffix = """Please follow the above format to explicitly show your full reasoning chain."""
    # Prompt 后缀：输出完整推论链。

    # Combine summary memory and window memory into prompt
    # 将摘要内存和窗口内存集成到提示中
    agent_prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "summary", '"level"']  # Add custom input variables
        # 添加自定义输入变量
    )

    # Build an LLMChain with the model and prompt
    # 使用 LLM 和提示构建推理链
    llm_chain = LLMChain(
        llm=base_llm,
        prompt=agent_prompt
    )

    # ---------- (E) Create the ZeroShotAgent ----------
    # ---------- (E) 创建零次学习代理 ----------
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools
    )

    # ---------- (F) Wrap into an AgentExecutor ----------
    # ---------- (F) 包装为 AgentExecutor ----------
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=window_memory,  # 绑定窗口内存
        verbose=True  # 输出交互过程
    )

    return agent_chain, summary_memory  # 返回代理链和摘要模块


# ===================== 8. 测试运行 =====================
# ===================== 8. Testing the RAG Agent =====================
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    index_path = "../Database/index.faiss"
    openai_api_key = "sk-proj-y-XpEu8YnMzAjPXX123456"  # 示例Key，请替换为实际API密钥

    # Initialize the Agent and memory modules
    # 初始化 Agent 和内存模块
    result = build_advanced_rag_agent_chain_of_thought(
        index_path=index_path,
        openai_api_key=openai_api_key,
        temperature=0.2,
        window_size=3  # 设置窗口对话阈值
    )

    # Unpack the result and handle exceptions
    # 解包结果并处理异常
    if len(result) == 2:  # Return agent_chain and window_memory only
        agent_chain, window_memory = result
        summary_memory = None
    elif len(result) == 3:  # Return agent_chain, window_memory, and summary_memory
        agent_chain, window_memory, summary_memory = result
    else:
        raise ValueError("Unexpected number of values returned by build_advanced_rag_agent_chain_of_thought")

    print("=== Welcome to the Advanced RAG Agent ===")
    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Conversation ended.")
            break

        # === (1) Save new user dialog into the window memory ===
        # === (1) 将新用户对话存入窗口内存 ===
        window_memory.chat_memory.add_message({"role": "user", "content": user_input})

        # === (2) Summarize older messages if window size is exceeded ===
        # === (2) 如果超过窗口大小则生成摘要 ===
        if len(window_memory.chat_memory.messages) >= 6:
            try:
                old_messages = window_memory.chat_memory.messages[:-2]
                summarized_input = " ".join(m["content"] for m in old_messages if m["role"] == "user")
                summarized_response = " ".join(m["content"] for m in old_messages if m["role"] == "agent")

                if summary_memory:
                    summary_memory.save_context(
                        {"input": summarized_input},
                        {"response": summarized_response}
                    )
                    logging.info("Summary saved successfully!")
                else:
                    logging.warning("Summary memory is inactive, skipping summary saving.")

                # 清理旧会话，仅保留最近两条
                window_memory.chat_memory.messages = window_memory.chat_memory.messages[-2:]
            except Exception as e:
                logging.error(f"Error during summary creation: {e}")

        # === (3) Load summary if available ===
        # === (3) 从摘要模块加载摘要 ===
        summary = summary_memory.get("summary", "无摘要") if summary_memory else "无摘要"

        # === (4) Build input data and invoke agent ===
        # === (4) 构建输入并调用代理 ===
        input_params = {
            "input": user_input,
            "chat_history": window_memory.chat_memory.messages[-2:],
            "summary": summary,
            '"level"': "medium"
        }

        logging.info(f"Input Parameters: {input_params}")

        try:
            response = agent_chain.invoke(input_params)
            window_memory.chat_memory.add_message({"role": "agent", "content": response})
            print(f"\nAI Agent Output:\n{response}\n")
        except Exception as e:
            logging.error(f"Error invoking Agent: {e}")
            print("\nSystem Message: An error occurred. Please retry or check the logs.\n")

