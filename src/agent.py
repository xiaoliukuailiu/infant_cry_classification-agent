"""
ReAct Agent — 婴儿哭声分析
"""
import json
import numpy as np
import torch
from openai import OpenAI
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS
from .features import preprocess
from .model import load_model, CryClassifier


class CryAnalysisAgent:
    """基于 ReAct 模式的哭声分析 Agent"""

    def __init__(self, config, model_path: str):
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        self.model_name = config.llm_model
        self.max_rounds = config.max_rounds
        self.class_names = config.class_names

        # 加载模型
        self.classifier = load_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

    def analyze(self, filepath: str) -> str:
        """上传音频 → Agent 推理 → 返回自然语言解释"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析这段婴儿哭声音频：{filepath}"},
        ]

        for _ in range(self.max_rounds):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content

            self.messages.append(msg)

            for tc in msg.tool_calls:
                if tc.function.name == "analyze_cry_audio":
                    args = json.loads(tc.function.arguments)
                    result = self._infer(args["filepath"])
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

        return self._force_answer()

    def _infer(self, filepath: str) -> str:
        """本地模型推理"""
        try:
            x = preprocess(filepath)
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits = self.classifier(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            top2_idx = np.argsort(probs)[::-1][:2]

            lines = []
            for rank, idx in enumerate(top2_idx, 1):
                cls_name = self.class_names[idx]
                conf = probs[idx]
                lines.append(f"  Top-{rank}: {cls_name}（置信度 {conf:.1%}）")

            return "模型推理结果：\n" + "\n".join(lines)

        except Exception as e:
            return f"推理失败: {e}"

    def _force_answer(self) -> str:
        self.messages.append({
            "role": "user",
            "content": "请基于以上分析结果给出最终判断，如果信息不足请诚实说明。",
        })
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        return response.choices[0].message.content
