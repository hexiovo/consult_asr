using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("咨询录音转写")]
[assembly: AssemblyDescription("本地双人中文咨询录音转写")]
[assembly: AssemblyProduct("ConsultASR")]
[assembly: AssemblyVersion("2.5.0.0")]
[assembly: AssemblyFileVersion("2.5.0.0")]

internal static class Launcher
{
    [STAThread]
    private static int Main(string[] args)
    {
        bool startupCheck = args.Length == 1 && args[0] == "--check-startup";
        bool nativeCheck = startupCheck || (args.Length == 1 && args[0] == "--check-native");
        bool check = nativeCheck || (args.Length == 1 && args[0] == "--check");
        try
        {
            string[] config = ReadResource("ConsultASR.Config").Trim().Split('\n');
            string project = config[0].Trim();
            string python = config[1].Trim();
            string ffmpeg = config.Length > 2 ? config[2].Trim() : "";
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            bool portable = config.Length > 3 && config[3].Trim() == "portable";
            if (portable)
            {
                project = Path.GetFullPath(Path.Combine(baseDirectory, project));
                python = Path.GetFullPath(Path.Combine(baseDirectory, python));
                ffmpeg = Path.GetFullPath(Path.Combine(baseDirectory, ffmpeg));
            }
            DirectoryInfo parent = Directory.GetParent(AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/'));
            if (!portable && parent != null)
            {
                string sibling = Path.Combine(parent.FullName, "consult_asr");
                if (File.Exists(Path.Combine(sibling, "consult_asr.py"))) project = sibling;
            }
            string entry = Path.Combine(project, "consult_asr.py");
            if (!File.Exists(python) || !File.Exists(entry))
                return Fail(check, "启动所需的程序目录或 Python 环境不存在。\n发布版请先运行 setup.bat 完成部署，并保留整个 ConsultASR 文件夹。", 2);

            string arguments = "-X utf8 -B \"" + entry + "\"";
            if (check)
            {
                string probe = Convert.ToBase64String(Encoding.UTF8.GetBytes(ReadResource("ConsultASR.Probe")));
                arguments = "-X utf8 -B -c \"import base64;exec(base64.b64decode('" + probe + "'))\"";
            }
            ProcessStartInfo start = new ProcessStartInfo(python, arguments);
            start.WorkingDirectory = project;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WindowStyle = ProcessWindowStyle.Hidden;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            string prefix = Path.GetDirectoryName(python);
            string extraPath = Path.GetDirectoryName(ffmpeg) + ";" + prefix + ";" + Path.Combine(prefix, "Scripts") + ";" +
                Path.Combine(prefix, "Library", "bin") + ";" +
                Path.Combine(prefix, "Lib", "site-packages", "torch", "lib");
            start.EnvironmentVariables["PATH"] = extraPath + ";" + (start.EnvironmentVariables["PATH"] ?? "");
            start.EnvironmentVariables["CONDA_PREFIX"] = prefix;
            start.EnvironmentVariables.Remove("PYTHONHOME");
            start.EnvironmentVariables.Remove("PYTHONPATH");
            start.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            start.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
            start.EnvironmentVariables["TRANSFORMERS_OFFLINE"] = "1";
            start.EnvironmentVariables["HF_HUB_DISABLE_TELEMETRY"] = "1";
            start.EnvironmentVariables["CONSULT_ASR_LAUNCHER_PATH"] = Assembly.GetExecutingAssembly().Location;
            if (File.Exists(ffmpeg) && String.IsNullOrWhiteSpace(start.EnvironmentVariables["CONSULT_ASR_FFMPEG"]))
                start.EnvironmentVariables["CONSULT_ASR_FFMPEG"] = ffmpeg;
            if (check) start.EnvironmentVariables["QT_QPA_PLATFORM"] = nativeCheck ? "windows" : "offscreen";

            bool displayStartup = !check || startupCheck;
            string eventName = "Local\\ConsultASR.Startup." + Guid.NewGuid().ToString("N");
            if (displayStartup)
            {
                start.EnvironmentVariables["CONSULT_ASR_STARTUP_EVENT"] = eventName;
                if (startupCheck) start.EnvironmentVariables["CONSULT_ASR_CHECK_STARTUP"] = "1";
                Application.EnableVisualStyles();
            }
            using (EventWaitHandle startupReady = new EventWaitHandle(false, EventResetMode.ManualReset, eventName))
            using (StartupWindow splash = new StartupWindow())
            using (Process process = new Process())
            {
                if (displayStartup)
                {
                    splash.Show();
                    splash.Refresh();
                }
                process.StartInfo = start;
                // Drain console streams without retaining or writing consultation content.
                process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { };
                process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { };
                if (!process.Start()) return Fail(check, "无法启动转写程序。", 3);
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                bool startupSignaled = false;
                if (displayStartup)
                {
                    using (System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer())
                    {
                        timer.Interval = 100;
                        timer.Tick += delegate
                        {
                            startupSignaled = startupReady.WaitOne(0);
                            if (startupSignaled || process.HasExited) splash.Close();
                        };
                        timer.Start();
                        Application.Run(splash);
                    }
                }
                process.WaitForExit();
                if (process.ExitCode != 0)
                    return Fail(check, "转写程序异常退出。\n请检查现有 Python 环境，或查看项目 log 文件夹中的诊断日志。", process.ExitCode);
                if (startupCheck && !startupSignaled)
                    return Fail(check, "启动窗口未收到主界面就绪通知。", 5);
            }
            return 0;
        }
        catch (Exception)
        {
            return Fail(check, "启动失败。请确认项目目录及本地运行环境可访问，再重新启动。", 4);
        }
    }

    private sealed class StartupWindow : Form
    {
        private readonly Icon appIcon;

        internal StartupWindow()
        {
            Text = "咨询录音转写 · 正在启动";
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedSingle;
            ControlBox = false;
            ShowInTaskbar = false;
            AutoScaleMode = AutoScaleMode.Dpi;
            ClientSize = new Size(440, 210);
            BackColor = Color.FromArgb(255, 254, 250);
            ForeColor = Color.FromArgb(36, 87, 76);
            Font = new Font("Microsoft YaHei UI", 10F);
            appIcon = Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location);
            Icon = appIcon;
            Label title = new Label();
            title.Text = "咨询录音转写";
            title.Font = new Font(Font.FontFamily, 18F, FontStyle.Bold);
            title.SetBounds(28, 26, 380, 42);
            Controls.Add(title);
            Label status = new Label();
            status.Text = "正在启动，请稍候…";
            status.SetBounds(30, 83, 380, 30);
            Controls.Add(status);
            ProgressBar progress = new ProgressBar();
            progress.Style = ProgressBarStyle.Marquee;
            progress.MarqueeAnimationSpeed = 25;
            progress.SetBounds(30, 128, 380, 8);
            Controls.Add(progress);
            Label detail = new Label();
            detail.Text = "正在准备界面与本地资源";
            detail.ForeColor = Color.FromArgb(82, 100, 90);
            detail.SetBounds(30, 153, 380, 28);
            Controls.Add(detail);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (disposing && appIcon != null) appIcon.Dispose();
        }
    }

    private static string ReadResource(string name)
    {
        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name))
        using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
            return reader.ReadToEnd();
    }

    private static int Fail(bool check, string message, int code)
    {
        if (!check) MessageBox.Show(message, "咨询录音转写", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        return code;
    }
}
