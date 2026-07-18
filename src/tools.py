"""
Agent 工具定义 — Function Calling
"""
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_cry_audio",
            "description": (
                "分析上传的婴儿哭声音频文件。提取 MFCC 特征，"
                "使用深度学习模型推理，返回 Top-2 预测类别及置信度。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "音频文件的路径",
                    },
                },
                "required": ["filepath"],
            },
        },
    },
]
