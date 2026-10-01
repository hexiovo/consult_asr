# 咨询录音转写

Windows 本地桌面工具：使用 faster-whisper large-v3 转写中文、使用 pyannote.audio 区分两位说话人，并可用 emotion2vec 展示语音情绪倾向和声学特征。保存前可预览文本并确认角色；音频与转写只在本机处理。

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
   python tools/prepare_models.py --tone
   python tools/prepare_models.py --verify
   ```

   工具将 faster-whisper large-v3 固定在 `edaa852ec7e145841d8ffdb056a99866b5f0a478`，并将模型保存在 `local_models/`。pyannote 配置引用的 segmentation、embedding 等关联模型也会下载到本地，并将引用改写为本地 `pytorch_model.bin` 检查点。额外的 `--tone` 会在独立的 `.runtime/tone` 依赖目标中准备 CPU 情绪推理依赖，并将 `iic/emotion2vec_plus_base` 固定到 revision `58354f1e5b6c118cf8b175b5b1955d6b07df42c2` 后下载到 `models/emotion2vec_plus_base/`；`model.pt` 的 SHA-256 为 `60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd`。不会改动 `requirements-lock.txt` 中已有的 Torch、NumPy、SciPy 和 pyannote 版本。界面开始处理前会确认所选分析模型已准备；运行阶段只读本地资源，不会自动联网下载。

   pyannote 的 `speaker-diarization-3.1` 和 `segmentation-3.0` 可能要求账号分别取得访问许可。请先在 Hugging Face 对提示的仓库完成账号授权；工具不会替用户接受条款。已有 ASR 权重后，双击 `prepare_models.bat` 即可只准备 pyannote 模型。令牌在控制台以隐藏方式输入，只在准备进程内存中使用，不进入命令行、环境变量、日志或磁盘。HTTP 401/403 会停止模型准备并说明受阻仓库，其他代码工作不受影响。

3. 运行 `run_consult_asr.bat`，或在激活环境后执行：

   ```powershell
   python consult_asr.py
   ```

## 使用

- 在左侧添加音频文件或文件夹，队列会显示每条录音的处理状态；选择一条录音后，可在右侧查看逐字稿、说话人示例并确认角色。支持 WAV、MP3、M4A、FLAC、AAC、OGG、WMA 和 MP4。
- 窗口底部始终显示结果保存目录，可直接编辑或通过“更改…”选择目录。设备、精度和 beam size 等识别设置默认收起，需要时点击“显示高级识别设置”。
- 开始处理前，large-v3 与 pyannote 双人区分模型都必须就绪；未就绪时“开始处理”保持禁用。
- 情绪倾向分析默认启用；使用前运行 `python tools/prepare_models.py --tone`。未准备情绪模型时可取消勾选“情绪倾向分析”，这只会关闭情绪模型；程序仍会转写并为说话人归属安全的窗口提取声学特征。
- 每个文件单独完成规范化、识别和说话人区分，并提取声学特征；启用情绪倾向后，再使用同一份临时规范化音频做本地 CPU 分析。批量文件按队列顺序处理。处理中可请求安全取消，当前模型阶段会在安全边界停止。
- 预览区显示逐字稿和两位说话人的示例发言。默认以“说话人1/说话人2”保存；只有明确选择并保存“说话人1/2是咨询师”后，导出才使用“咨询师/来访者”。
- 启用情绪分析后，程序在规范化音频仍位于任务临时目录时，按不超过 6 秒的可靠说话人窗口计算情绪倾向。窗口须有唯一说话人轨道覆盖；局部转写不确定、说话人未归属、说话人重叠、轨道缺失或与词归属不一致时显示“未知”。没有 diarization 轨道覆盖的词间隙也标为未知，不把声学或情绪结果归给附近说话人。情绪标签是通用模型输出，不是诊断；显示的模型置信分数是未校准原始分数，不代表咨询场景准确率或概率。
- 声学特征也按相同的可靠窗口显示中文字符语速、RMS 音量、段内停顿以及基频和句末音高走势。语速是根据安全窗口内的 ASR 中文字符估算，界面和 JSON 同时列出字符数和分母时长；跨窗口的词只计一次。句末上扬只作为声学候选，不据此判定说话内容是不是疑问句。
- Word 会把相邻的同一说话人发言合并，并按窗口列出声学和情绪结果。JSON 同时保留每个识别片段、合并后的段落、窗口级声学结果、逐窗情绪原始分数、模型版本、说话人、角色确认状态及待核对标记。含声学或情绪扩展字段的文件使用 schema version 2；未启用扩展分析的旧调用仍输出 version 1。
- 同名结果不会覆盖；新文件会使用带序号的文件名，Word 与 JSON 始终成对保存。再次保存时只处理未导出或角色设置变更的录音。
- 无语音文件仍可导出一份明确标记“未检测到语音”的 Word 与 JSON。识别失败和导出失败均按文件显示，不会报告为成功。

## 本地资源与离线运行

- 默认模型目录：`local_models/faster-whisper-large-v3`、`local_models/pyannote/speaker-diarization-3.1`、`local_models/openai-whisper` 和 `models/emotion2vec_plus_base`；独立运行依赖写入 `.runtime/tone`。模型/运行时路径在 Git 中忽略。设置 `CONSULT_ASR_MODEL_DIR` 可切换 ASR 与 pyannote 模型根目录；情绪资源使用固定项目目录。
- FFmpeg 优先取 `CONSULT_ASR_FFMPEG`，其次检查当前进程 PATH、当前 Conda 环境及 Conda 环境清单中的项目环境。若无法自动发现，可在当前 PowerShell 会话设置 `CONSULT_ASR_FFMPEG` 指向 `ffmpeg.exe`。程序只规范化到任务专属临时目录，不改写源音频。
- 模型准备依赖网络；模型齐全后，ASR、说话人区分与情绪识别在离线模式运行。情绪识别通过当前锁定 Python 调用隔离 runtime，固定使用 CPU；模型或关联检查点未准备好时，默认情绪分析会禁用“开始处理”，不会自动请求下载。
- 2.0.3 基线曾验证 pyannote 本地权重和 CPU pipeline 初始化；该结果不包含说话人分离音频 forward、实际录音或 GPU 推理。2.2.0 的固定 emotion2vec revision 与隔离运行时通过离线 CPU forward 和端到端 pipeline/成对导出冒烟检查，合成单声道音频返回了 9 个有限原始分数及匹配的模型 revision、权重哈希和 CPU 元数据。当前 `--verify` 检查通过；这些检查验证本地运行链路，不代表情绪分类准确率。
- 本地 manifest 记录三个 pyannote 仓库各自解析出的 40 位 revision。准备工具没有指定固定 pyannote revision，清单值反映这些本地模型实际对应的版本。
- 质量优先默认 beam 5 与 float16；CPU 自动改用兼容精度，速度选项使用 int8_float16（CPU 上使用 int8）。每次只处理一个音频文件，并在 ASR 和说话人区分阶段间释放模型。

## 结果边界

没有用户录音和人工校对稿可用于测量字错率，因此当前交付不承诺识别准确率，也不宣称一小时录音性能通过。短合成音频只用于验证其实际覆盖的处理路径。双人分离结果可能有不确定片段，导出会保留并标注，供人工核对。
