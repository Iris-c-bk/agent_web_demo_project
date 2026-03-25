# app.py
import sys
from pathlib import Path

_AI_ROOT = Path(__file__).resolve().parent.parent
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from env_config import load_api_key, load_tavily_api_key

import streamlit as st
from agent_core_history import SmartAgent  # 带 SQLite 持久化的版本（load/save_history）
# ================= 配置区域 =================
# SiliconFlow：SILICONFLOW_API_KEY 或 OPENAI_API_KEY，或本目录 .env
# Tavily：TAVILY_API_KEY，或同上 .env

# LLM（SiliconFlow 兼容 OpenAI 接口）；换服务商时同步改 MODEL_NAME
_api_key = load_api_key()
_api_base_url = "https://api.siliconflow.cn/v1"
# Tavily Key（搜索走 _tavily_search_http，避免在 Python 3.8 下导入 tavily 包报错）
_tavily_api_key = load_tavily_api_key()
_model_name = "deepseek-ai/DeepSeek-V3"

# 页面配置
st.set_page_config(page_title="我的超级 Agent", page_icon="🤖")
st.title("🤖 我的私人智能助手 (带记忆 + 联网)")
st.title("记忆存储位置：memory.db")

# 侧边栏：配置与控制
with st.sidebar:
    st.header("设置")
    api_key = st.text_input("API Key", type="password", value="sk-...") # 可预设
    session_id = st.text_input("用户ID", value="user_01")
    
    if st.button("🗑️ 清空记忆"):
        st.session_state.agent.clear_history()
        st.session_state.agent.save_history(session_id)
        st.rerun()
    
    st.markdown("---")
    st.info("💡 提示：试着问我今天的新闻，或者让我帮你规划旅行。")

# 初始化 Agent (利用 Streamlit 的 session_state 保持状态)
if "agent" not in st.session_state:
    # 这里填入你真实的 Key 和配置，或者从环境变量读取
    st.session_state.agent = SmartAgent(
        api_key= _api_key, 
        base_url= _api_base_url,
        model_name= _model_name,
        tavily_key= _tavily_api_key
    )
    # 尝试加载历史
    st.session_state.agent.load_history(session_id)

# 显示历史聊天记录（忽略 tool/system 等无可见文本的消息）
for msg in st.session_state.agent.get_history():
    role = msg.get("role")
    if role not in ("user", "assistant"):
        continue

    content = msg.get("content")
    if not content:
        continue

    with st.chat_message(role):
        st.write(content)

# 聊天输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
    
    # 2. 显示思考状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 正在思考并搜索...")
        # 3. 调用核心逻辑
        try:
            response, thought_process = st.session_state.agent.chat(prompt)
            # 保存记忆
            st.session_state.agent.save_history(session_id)
            
            # 4. 显示最终回复
            message_placeholder.markdown(response)
            if thought_process:
                with st.expander("🔍 查看思考过程"):
                    for item in thought_process:
                        st.markdown(
                            "### Step {step}: `{tool}`".format(
                                step=item.get("step", "-"),
                                tool=item.get("tool", "unknown"),
                            )
                        )
                        st.write("参数：", item.get("args", {}))
                        st.write("结果：")
                        st.code(str(item.get("result", "")))
            else:
                with st.expander("🔍 查看思考过程"):
                    st.caption("本轮没有发生工具调用。")
        except Exception as e:
            message_placeholder.error(f"出错了: {e}")