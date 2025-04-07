import gradio as gr
import os
from backend.Main_fn import Ask

# 确保CSS和JS目录存在
# Ensure CSS and JS directories exist
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# 设置Gradio界面参数
# Set up Gradio interface parameters
title = "NBPS Assistance"
description = "NBPS校园问答助手 (Chinese)\nCampus Q&A Helper (English)"
article = "Demo版本 不代表最终质量 (Chinese)\nDemo version does not represent final quality (English)"
share = True

# 自定义HTML头，包含设备检测脚本
# Custom HTML head with device detection script
custom_head = """
<script src="/static/js/device_detection.js"></script>
<style>
    body {
        font-family: Arial, sans-serif;
        transition: all 0.3s ease;
    }
    
    /* 适用于所有设备/浏览器的基本样式 */
    /* Base styles for all devices/browsers */
    .gradio-container {
        transition: max-width 0.3s ease, padding 0.3s ease;
    }
</style>
"""

def gradio_ask(user_input):
    # 调用Ask函数并返回结果
    # Call Ask function and return the result
    return Ask(user_input)

# 创建响应式Gradio界面
# Create responsive Gradio interface
iface = gr.Interface(
    fn=gradio_ask,
    inputs=gr.Textbox(placeholder="在此输入您的问题...", elem_classes="input-box"),
    outputs=gr.Textbox(elem_classes="output-box"),
    title=title,
    description=description,
    article=article,
    css=custom_head,
    allow_flagging=False
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=share)
