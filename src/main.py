"""
CLI 入口 — 训练、分析、Web UI 一键启动
"""
import os
import click
from pathlib import Path
from .config import config
from .agent import CryAnalysisAgent


@click.group()
def cli():
    """婴儿哭声分类 Agent — 上传音频，AI 自动推理哭声原因"""


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
def train_cmd(data_dir):
    """训练分类模型

    DATA_DIR: 训练数据目录，结构需为 data/train/<类别>/xxx.wav
    """
    from .train import train
    train(data_dir)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", "-m", default="./models/cry_model_best.pt", help="模型路径")
def analyze(audio_file, model):
    """分析一段婴儿哭声音频"""
    if not os.path.exists(model):
        click.secho(f"模型不存在: {model}", fg="red")
        click.echo("请先训练: cry-agent train ./data/train/")
        return

    click.echo("加载模型...")
    agent = CryAnalysisAgent(config, model)

    click.echo("分析中...")
    result = agent.analyze(audio_file)
    click.echo(f"\n{result}")


@cli.command()
@click.option("--model", "-m", default="./models/cry_model_best.pt", help="模型路径")
@click.option("--port", "-p", default=8501, help="端口号")
def ui(model, port):
    """启动 Streamlit Web 界面"""
    import sys
    import subprocess

    ui_path = Path(__file__).parent / "ui" / "app.py"
    if not ui_path.exists():
        click.secho("UI 文件不存在", fg="red")
        return

    env = os.environ.copy()
    env["CRY_MODEL_PATH"] = os.path.abspath(model)
    env["DEEPSEEK_API_KEY"] = config.llm_api_key
    env["OPENAI_API_KEY"] = config.llm_api_key  # OpenAI SDK 也会检查这个

    click.echo(f"启动 Web UI: http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path),
         "--server.port", str(port)],
        env=env,
    )


if __name__ == "__main__":
    cli()
