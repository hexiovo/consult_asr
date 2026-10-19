# Windows 下载与部署

适用于 Windows 10/11 64 位。无需预装 Python、Conda、FFmpeg 或 CUDA Toolkit，无需管理员权限。支持 CPU；NVIDIA GPU 需要兼容 CUDA 12.4 的显卡驱动（建议使用当前稳定驱动）。内存建议至少 16 GB，首次部署建议预留 30 GB 以上空间。公开模型与运行环境的下载量以 Release 附件为准。

## 在其他电脑使用

1. 打开 <https://github.com/hexiovo/consult_asr/releases/latest>，下载 `ConsultASR-2.5.0-Windows-x64-Setup.zip`。
2. 解压到可写目录（例如 `D:\Apps\ConsultASR-Setup`），双击 `setup.bat`。不要在 ZIP 预览窗口内直接运行。
3. 部署工具自动下载同版本的应用、Python 运行环境、FFmpeg、ASR 与情绪模型，进行 SHA-256 校验并解压。大文件会分卷下载；网络中断后重新运行 `setup.bat` 会复用已下载文件并续传。默认安装在部署工具旁的 `ConsultASR` 文件夹中。
4. 首次部署时，按提示用自己的 Hugging Face 账户分别取得以下模型的访问许可，再创建具有对应仓库读取权限的令牌，在控制台隐藏输入。令牌不保存。程序不会代替你接受模型访问条件。

   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - 令牌页面：<https://huggingface.co/settings/tokens>

5. 完成后进入 `ConsultASR`，双击 `咨询录音转写.exe`。后续转录离线运行。也可双击 `run_consult_asr.bat`。请保留整个文件夹。

若模型授权或下载未完成，可在完成授权后重新运行同一 `setup.bat`，或运行安装目录中的 `prepare_models.bat`；无须重新下载运行环境。未准备好必需的双人区分模型时，界面会禁用“开始处理”。

## 下载、移动与更新

- 可以先在另一台电脑下载所有 Release 数据附件，将它们与解压后的 `setup.bat`、`setup.ps1`、`release-manifest.json` 放在同一目录；部署时优先校验和使用本地附件。pyannote 模型仍需按其访问要求单独准备。
- 不用手工合并 `.zip.001` 等分卷，直接运行 `setup.bat`。
- 自定义位置：在部署工具目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -InstallDirectory D:\Apps\ConsultASR-2.5.0`。新版本请安装到新目录，已有其他版本或非空目录不会被覆盖。
- 安装完成后可移动整个应用文件夹。应用、模型和启动器使用相对布局；应用只通过自带 Python 直接启动，移动后不依赖原来的 Conda 路径。不要在移动后调用运行环境中面向开发者的 pip/Conda 控制台脚本。
- 运行并确认正常后，可以删除部署工具目录中的 `.downloads` 缓存，回收下载占用空间。不要删除应用中的 `.runtime`、`local_models` 和 `models`。
- `diagnose.bat` 执行离线依赖导入、FFmpeg 合成静音转换和模型文件检查。它不会读取咨询录音，也不代表实际识别准确率已经验证。输出中可能包含本机路径，分享前自行检查。
- 首次下载的部署脚本和 EXE 未做商业代码签名。若 Windows 提示来源未知，请核对下载仓库及 Release 中的 `SHA256SUMS.txt`。

## 从源码重新构建 Release

在构建主机上保留已验证的 Python 3.10 环境、公开 ASR/情绪模型和 `.runtime/tone`。构建工具使用 `conda-pack` 和 `pefile`；目标电脑不需要它们。FFmpeg 参数应指向带完整相邻 DLL 和许可文件的已部署 Conda FFmpeg。

```powershell
python packaging/build_release.py --python-prefix C:\Envs\consult_asr_fast --ffmpeg C:\Envs\audio\Library\bin\ffmpeg.exe
```

输出位于 `dist/v2.5.0/`，包含安装器 ZIP、分卷数据、版本清单与校验和。构建过程不收集日志、录音、转录、用户令牌、会话缓存或受访问限制的 pyannote 权重。Python 运行环境保留第三方许可及发行包元数据；FFmpeg 随包保留其许可证和源码地址。依赖锁定清单用于维护与诊断，最终用户直接使用随包的已验证运行环境。
