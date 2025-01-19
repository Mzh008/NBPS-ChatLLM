#Main_fn.py


def Ask(user_input: str):
    """
    对特定用户的输入(user_input)进行处理，调用AI Agent并返回结果。
    Process the user's input (user_input), invoke the AI Agent, and return the response.
    """
    import logging
    from Agent.ai_agent import build_advanced_rag_agent_chain_of_thought

    # 初始化日志记录模块
    # Initialize the logging module
    logging.basicConfig(level=logging.INFO)

    # 设置索引文件路径和OpenAI API密钥
    # Set the FAISS index file path and OpenAI API key
    index_path = "../Database/index.faiss"
    openai_api_key = "sk-proj-y-XpEu8YnMzAjPXX123456"  # 示例key，需替换为实际API密钥 / Example key, replace with your real API key

    # === 初始化Agent和记忆模块 ===
    # === Initialize Agent and memory module ===
    result = build_advanced_rag_agent_chain_of_thought(
        index_path=index_path,
        openai_api_key=openai_api_key,
        temperature=0.2,  # 控制生成文本的随机性 (越低越确定) / Control randomness of generated text (lower is more deterministic)
        window_size=3  # 设置窗口对话的阈值 / Set the threshold for the conversation window size
    )

    # === 解包返回结果并处理异常情况 ===
    # === Unpack the returned result and handle exceptions ===
    if len(result) == 2:  # 如果返回2个值，仅为 agent_chain 和 window_memory
        # If two values are returned, assume these are agent_chain and window_memory
        agent_chain, window_memory = result
        summary_memory = None  # 设置摘要为默认值 None / Set summary_memory to default None
    elif len(result) == 3:  # 如果返回3个值，为 agent_chain、window_memory、summary_memory
        # If three values are returned, these are agent_chain, window_memory, and summary_memory
        agent_chain, window_memory, summary_memory = result
    else:
        # 如果返回的值数量异常，则抛出错误 / Raise an error for unexpected result
        raise ValueError("Unexpected number of values returned by build_advanced_rag_agent_chain_of_thought")

    # === (1) 将用户输入存入窗口记忆 ===
    # === (1) Add the user's input to the chat window memory ===
    window_memory.chat_memory.add_message({"role": "user", "content": user_input})

    # === (2) 检查是否需要生成对话摘要 ===
    # === (2) Check whether to generate a conversation summary ===
    if len(window_memory.chat_memory.messages) >= 6:  # 检查窗口对话大小是否超出阈值 / Check if the window size exceeds the threshold
        try:
            # 提取旧对话消息（保留最新两条消息）
            # Extract old dialog messages (keep the latest two)
            old_messages = window_memory.chat_memory.messages[:-2]
            summarized_input = " ".join(m["content"] for m in old_messages if m["role"] == "user")  # 用户旧消息
            summarized_response = " ".join(m["content"] for m in old_messages if m["role"] == "agent")  # 代理旧消息

            # 如果摘要记忆模块可用，则保存摘要
            # Save the summary if the summary memory module is available
            if summary_memory:
                summary_memory.save_context(
                    {"input": summarized_input},  # 摘要的用户输入 / Summarized user input
                    {"response": summarized_response}  # 摘要的代理响应 / Summarized agent response
                )
                logging.info("摘要保存成功：用户输入和响应已被截断。")  # Successfully saved summary
            else:
                logging.warning(
                    "摘要记忆模块（summary_memory）未激活，跳过摘要保存。")  # Warning if summary memory is not active

            # 删除旧的对话，仅保留最近两条
            # Remove old dialogs and keep only the last two messages
            window_memory.chat_memory.messages = window_memory.chat_memory.messages[-2:]
        except Exception as e:
            # 如果摘要创建失败，记录错误日志 / Log an error if summary creation fails
            logging.error(f"摘要创建过程出现错误：{e}")

    # === (3) 从摘要记忆模块加载摘要并设置默认值 ===
    # === (3) Load summaries from the summary memory module and set default values ===
    if summary_memory:  # 如果摘要记忆模块可用 / If the summary memory module is available
        summary = summary_memory.get("summary", "无摘要")  # 使用默认摘要为 "无摘要" / Use "无摘要" (No Summary) as default
    else:  # 如果记忆不可用，则摘要设置为默认值
        # If memory is unavailable, set summary to default
        summary = "无摘要"

    # === (4) 构建输入参数，并调用Agent ===
    # === (4) Construct input parameters and invoke the Agent ===
    input_params = {
        "input": user_input,  # 用户输入 / User input
        "chat_history": window_memory.chat_memory.messages[-2:],  # 对话历史（最近两条消息） / Chat history (last two messages)
        "summary": summary,  # 当前摘要 / Current summary
        "level": "medium"  # 确保包含优先级字段，设置为默认值 "medium" / Ensure priority level is included, set to "medium"
    }

    logging.info(f"输入参数: {input_params}")  # 记录输入日志 / Log the input parameters

    try:
        # 调用Agent处理输入
        # Invoke the Agent to process the input
        response = agent_chain.invoke(input_params)  # 替换了run方法，执行invoke任务 / Use the invoke function instead of run

        # 将Agent的响应存储到窗口记忆
        # Store the Agent's response in the chat window memory
        window_memory.chat_memory.add_message({"role": "agent", "content": response})

        # 返回Agent的响应给用户
        # Return the Agent's response to the user
        return f"{response}"
    except Exception as e:
        # 遇到运行错误时记录日志并返回提示
        # Log errors and return a user-friendly message on failure
        logging.error(f"调用Agent时出错: {e}")
        return "\n系统提示：运行时出错，请重试或检查日志。\n"  # System message informing runtime errors
