# 咨询录音转写

Windows 本地桌面工具：使用 faster-whisper large-v3 转写中文、使用 pyannote.audio 区分两位说话人，并在保存前提供文本预览和角色确认。音频与转写只在本机处理。

## 启动

1. 创建独立的 `consult_asr_fast` Python 3.10 环境，并按锁定清单安装依赖：

   ```powershell
   conda create -n consult_asr_fast python=3.10 pip -c defaults
   conda activate consult_asr_fast
   python -m pip install -r requirements-lock.txt
   ```

   旧的 `consult_asr` 环境保持不变。`requirements-lock.txt` 收录验证环境的 110 项 `pip freeze`，并显式固定 `setuptools==80.9.0`，共 111 项精确依赖。该兼容版本为 pyannote.audio 3.1.1 提供 `pkg_resources`。隔离环境的 `pip check` 与 pyannote.audio 导入检查均通过。依赖从官方 PyPI 与 PyTorch HTTPS 源安装；CUDA 元数据可用不等于模型推理已验证。

2. 准备本地模型：

   ```powershell
   python tools/prepare_models.py
   python tools/prepare_models.py --verify
   ```

   工具将 faster-whisper large-v3 固定在 `edaa852ec7e145841d8ffdb056a99866b5f0a478`，并将模型保存在 `local_models/`。pyannote 配置引用的 segmentation、embedding 等关联模型也会下载到本地，并将引用改写为本地 `pytorch_model.bin` 检查点。界面开始处理前会确认 pipeline 配置结构有效，并检查 segmentation、embedding 都指向模型树内非空的绝对 checkpoint 路径；推理时强制使用本地资源，不会自动联网下载。

   pyannote 的 `speaker-diarization-3.1` 和 `segmentation-3.0` 可能要求账号分别取得访问许可。请先在 Hugging Face 对提示的仓库完成账号授权；工具不会替用户接受条款。已有 ASR 权重后，双击 `prepare_models.bat` 即可只准备 pyannote 模型。令牌在控制台以隐藏方式输入，只在准备进程内存中使用，不进入命令行、环境变量、日志或磁盘。HTTP 401/403 会停止模型准备并说明受阻仓库，其他代码工作不受影响。

3. 运行 `run_consult_asr.bat`，或在激活环境后执行：

   ```powershell
   python consult_asr.py
   ```

## 使用

- 选择一个音频文件或一个文件夹，再选择结果目录。支持 WAV、MP3、M4A、FLAC、AAC、OGG、WMA 和 MP4。
- 每个文件单独完成规范化、识别和说话人区分；批量文件按队列顺序处理。处理中可请求安全取消，当前模型阶段会在安全边界停止。
- 预览区显示逐字稿和两位说话人的示例发言。默认以“说话人1/说话人2”保存；只有明确选择并保存“说话人1/2是咨询师”后，导出才使用“咨询师/来访者”。
- Word 会把相邻的同一说话人发言合并。JSON 同时保留每个识别片段和合并后的段落时间、说话人、角色确认状态及待核对标记。
- 同名结果不会覆盖；新文件会使用带序号的文件名，Word 与 JSON 始终成对保存。再次保存时只处理未导出或角色设置变更的录音。
- 无语音文件仍可导出一份明确标记“未检测到语音”的 Word 与 JSON。识别失败和导出失败均按文件显示，不会报告为成功。

## 本地资源与离线运行

- 默认模型目录：`local_models/faster-whisper-large-v3`、`local_models/pyannote/speaker-diarization-3.1` 和 `local_models/openai-whisper`。设置 `CONSULT_ASR_MODEL_DIR` 可切换统一的模型根目录。
- FFmpeg 优先取 `CONSULT_ASR_FFMPEG`，其次检查当前进程 PATH、当前 Conda 环境及 Conda 环境清单中的项目环境。若无法自动发现，可在当前 PowerShell 会话设置 `CONSULT_ASR_FFMPEG` 指向 `ffmpeg.exe`。程序只规范化到任务专属临时目录，不改写源音频。
- 模型准备依赖网络；模型齐全后，识别和说话人区分在离线模式运行。模型或关联检查点未准备好时，界面会禁用“开始处理”，不会自动请求下载。
- 当前工作副本的 `python tools/prepare_models.py --verify` 于 2026-09-30 通过；本地 `config.yaml` 的 segmentation 和 embedding 绝对 checkpoint 路径均存在。真实加载器 `core.diarizer.Diarizer(device="cpu").load()` 在 Hugging Face/Transformers 离线标记、`CUDA_VISIBLE_DEVICES=-1` 且未提供 token 时，成功初始化 `pyannote.audio 3.1.1` 的 `SpeakerDiarization`。这验证本地权重和 CPU pipeline 初始化，不包含音频 forward、实际录音或 GPU 推理；GPU 兼容验收仍待完成。
- 本地 manifest 记录三个 pyannote 仓库各自解析出的 40 位 revision。准备工具没有指定固定 pyannote revision，清单值反映这些本地模型实际对应的版本。
- 质量优先默认 beam 5 与 float16；CPU 自动改用兼容精度，速度选项使用 int8_float16（CPU 上使用 int8）。每次只处理一个音频文件，并在 ASR 和说话人区分阶段间释放模型。

## 结果边界

没有用户录音和人工校对稿可用于测量字错率，因此当前交付不承诺识别准确率，也不宣称一小时录音性能通过。短合成音频只用于验证其实际覆盖的处理路径。双人分离结果可能有不确定片段，导出会保留并标注，供人工核对。
