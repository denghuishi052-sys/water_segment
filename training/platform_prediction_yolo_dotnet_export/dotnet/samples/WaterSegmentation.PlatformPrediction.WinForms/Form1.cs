using System.Diagnostics;
using WaterSegmentation.PlatformPrediction;

namespace WaterSegmentation.PlatformPrediction.WinForms;

public sealed class Form1 : Form
{
    private readonly TextBox _packageRoot = new();
    private readonly TextBox _imagePath = new();
    private readonly TextBox _outputDir = new();
    private readonly Button _runButton = new();
    private readonly PictureBox _inputPreview = new();
    private readonly PictureBox _overlayPreview = new();
    private readonly Label _status = new();

    public Form1()
    {
        Text = "Water Segmentation Demo";
        Width = 1180;
        Height = 760;
        MinimumSize = new Size(980, 620);
        StartPosition = FormStartPosition.CenterScreen;

        var defaultRoot = FindPackageRoot();
        _packageRoot.Text = defaultRoot;
        _imagePath.Text = Path.Combine(defaultRoot, "samples", "valid_water_0003.jpg");
        _outputDir.Text = Path.Combine(defaultRoot, "dotnet_runs", "gui_demo");

        Controls.Add(BuildLayout());
        LoadPreview(_inputPreview, _imagePath.Text);
    }

    private Control BuildLayout()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(12),
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 150));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));

        var inputs = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 3,
            RowCount = 4,
        };
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120));
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));

        AddPathRow(inputs, 0, "项目包目录", _packageRoot, BrowseFolder);
        AddPathRow(inputs, 1, "输入图片", _imagePath, BrowseImage);
        AddPathRow(inputs, 2, "输出目录", _outputDir, BrowseFolder);

        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight };
        _runButton.Text = "开始预测";
        _runButton.Width = 120;
        _runButton.Height = 34;
        _runButton.Click += async (_, _) => await RunPredictionAsync();
        buttonPanel.Controls.Add(_runButton);
        inputs.Controls.Add(buttonPanel, 1, 3);
        inputs.SetColumnSpan(buttonPanel, 2);

        var previews = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 2,
        };
        previews.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        previews.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        previews.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
        previews.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        previews.Controls.Add(new Label { Text = "原图", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, 0);
        previews.Controls.Add(new Label { Text = "预测叠加图", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 1, 0);
        ConfigurePictureBox(_inputPreview);
        ConfigurePictureBox(_overlayPreview);
        previews.Controls.Add(_inputPreview, 0, 1);
        previews.Controls.Add(_overlayPreview, 1, 1);

        _status.Dock = DockStyle.Fill;
        _status.TextAlign = ContentAlignment.MiddleLeft;
        _status.Text = "就绪";

        root.Controls.Add(inputs, 0, 0);
        root.Controls.Add(previews, 0, 1);
        root.Controls.Add(_status, 0, 2);
        return root;
    }

    private static void AddPathRow(
        TableLayoutPanel table,
        int row,
        string label,
        TextBox textBox,
        Action<TextBox> browseAction)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
        table.Controls.Add(new Label { Text = label, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, row);
        textBox.Dock = DockStyle.Fill;
        table.Controls.Add(textBox, 1, row);
        var button = new Button { Text = "浏览", Dock = DockStyle.Fill };
        button.Click += (_, _) => browseAction(textBox);
        table.Controls.Add(button, 2, row);
    }

    private static void ConfigurePictureBox(PictureBox pictureBox)
    {
        pictureBox.Dock = DockStyle.Fill;
        pictureBox.BorderStyle = BorderStyle.FixedSingle;
        pictureBox.BackColor = Color.FromArgb(32, 32, 32);
        pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
    }

    private static void BrowseFolder(TextBox target)
    {
        using var dialog = new FolderBrowserDialog { SelectedPath = Directory.Exists(target.Text) ? target.Text : "" };
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            target.Text = dialog.SelectedPath;
        }
    }

    private void BrowseImage(TextBox target)
    {
        using var dialog = new OpenFileDialog
        {
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff|All files|*.*",
            FileName = File.Exists(target.Text) ? target.Text : "",
        };
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            target.Text = dialog.FileName;
            LoadPreview(_inputPreview, dialog.FileName);
        }
    }

    private async Task RunPredictionAsync()
    {
        SetBusy(true, "正在预测...");
        try
        {
            var result = await Task.Run(() =>
            {
                var config = PredictionConfig.Load(Path.Combine(_packageRoot.Text, "configs", "dual_context.json"));
                using var service = new PredictionService(_packageRoot.Text, config);
                return service.PredictFile(_imagePath.Text, _outputDir.Text);
            });
            LoadPreview(_inputPreview, _imagePath.Text);
            LoadPreview(_overlayPreview, result.OverlayPath);
            _status.Text = $"完成：model={result.ModelType}, maskArea={result.MaskArea}, ratio={result.PredAreaRatio:F6}";
        }
        catch (Exception ex)
        {
            _status.Text = "失败：" + ex.Message;
            MessageBox.Show(this, ex.ToString(), "预测失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void SetBusy(bool busy, string? message = null)
    {
        _runButton.Enabled = !busy;
        Cursor = busy ? Cursors.WaitCursor : Cursors.Default;
        if (message is not null)
        {
            _status.Text = message;
        }
    }

    private static void LoadPreview(PictureBox box, string path)
    {
        if (!File.Exists(path))
        {
            return;
        }

        var old = box.Image;
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        box.Image = Image.FromStream(stream);
        old?.Dispose();
    }

    private static string FindPackageRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "configs", "yolo_only.json")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
    }
}
