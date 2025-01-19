# -*- coding: utf-8 -*-
#frontend/app.py
import gradio as gr
from backend.Main_fn import Ask

# 这里设置简单的Gradio界面参数（中文）
# Here we set up a simple Gradio interface (English)
title = "NBPS Assistance"
description = "NBPS校园问答助手 (Chinese)\nCampus Q&A Helper (English)"
article = "Demo版本 不代表最终质量 (Chinese)\nDemo version does not represent final quality (English)"
share = True

def gradio_ask(user_input):
    # 直接调用 Ask，并返回结果（中文）
    # Directly call Ask and return the result (English)
    return Ask(user_input)

iface = gr.Interface(
    fn=gradio_ask,
    inputs="text",
    outputs="text",
    title=title,
    description=description,
    article=article
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=share)