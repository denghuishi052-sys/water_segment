using System.Text.Json;
using System.Text.Json.Serialization;

namespace WaterSegmentation.PlatformPrediction;

public sealed class PredictionConfig
{
    public string ModelType { get; set; } = "yolo-onnx";
    public string ModelPath { get; set; } = "onnx/floodnet_binary_aug_yolov8m_1024.onnx";
    public PipelineConfig Pipeline { get; set; } = new();
    public Dictionary<string, JsonElement> ModelOptions { get; set; } = [];
    public Dictionary<string, JsonElement> PreprocessOptions { get; set; } = [];
    public Dictionary<string, JsonElement> PostprocessOptions { get; set; } = [];
    public Dictionary<string, JsonElement> OutputOptions { get; set; } = [];

    [JsonExtensionData]
    public Dictionary<string, JsonElement>? ExtensionData { get; set; }

    public int ImageSize { get; set; } = 1024;
    public float ConfidenceThreshold { get; set; } = 0.25f;
    public float IouThreshold { get; set; } = 0.5f;
    public float MaskThreshold { get; set; } = 0.65f;
    public float MaskBoxExpandRatio { get; set; } = 0.5f;
    public float MinAreaRatio { get; set; } = 0.0005f;
    public bool MorphClose { get; set; } = true;
    public bool SuppressLargeBorderComponents { get; set; } = false;
    public float LargeBorderComponentMinAreaRatio { get; set; } = 0.15f;
    public float LargeBorderComponentMarginRatio { get; set; } = 0.03f;
    public int MaxDetections { get; set; } = 300;
    public int ClassCount { get; set; } = 1;
    public int MaskPrototypeCount { get; set; } = 32;
    public CascadeConfig Cascade { get; set; } = new();
    public TilingConfig Tiling { get; set; } = new();
    public LargeImageConfig LargeImage { get; set; } = new();
    public string[] Providers { get; set; } = ["CUDAExecutionProvider", "CPUExecutionProvider"];

    public static PredictionConfig Load(string path)
    {
        var json = File.ReadAllText(path);
        var config = JsonSerializer.Deserialize<PredictionConfig>(
            json,
            new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                ReadCommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true
            });
        return config ?? throw new InvalidOperationException($"Invalid config: {path}");
    }

    public T GetModelOption<T>(string key, T defaultValue)
        => GetOption(ModelOptions, key, defaultValue);

    public T GetPreprocessOption<T>(string key, T defaultValue)
        => GetOption(PreprocessOptions, key, defaultValue);

    public T GetPostprocessOption<T>(string key, T defaultValue)
        => GetOption(PostprocessOptions, key, defaultValue);

    public T GetOutputOption<T>(string key, T defaultValue)
        => GetOption(OutputOptions, key, defaultValue);

    public static T GetOption<T>(
        IReadOnlyDictionary<string, JsonElement> options,
        string key,
        T defaultValue)
    {
        if (!options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return value.Deserialize<T>(
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }) ?? defaultValue;
        }
        catch (JsonException)
        {
            return defaultValue;
        }
    }
}

public sealed class PipelineConfig
{
    public string Type { get; set; } = "segmentation";
    public PipelineStepConfig[] Steps { get; set; } = [];
    public Dictionary<string, JsonElement> Options { get; set; } = [];
}

public sealed class PipelineStepConfig
{
    public string Name { get; set; } = "";
    public string Type { get; set; } = "";
    public bool Enabled { get; set; } = true;
    public Dictionary<string, JsonElement> Parameters { get; set; } = [];
}

public sealed class CascadeConfig
{
    public bool Enabled { get; set; } = true;
    public string[] ModelPaths { get; set; } = [];
    public int ImageSize { get; set; } = 640;
    public float ConfidenceThreshold { get; set; } = 0.15f;
    public float MinAspectRatio { get; set; } = 1.15f;
    public float TriggerAreaRatio { get; set; } = 0.005f;
    public float MinConsensusAreaRatio { get; set; } = 0.005f;
}

public sealed class TilingConfig
{
    public int TileSize { get; set; } = 1024;
    public int OverlapPixels { get; set; } = 256;
    public float Weight { get; set; } = 0.7f;
    public string StitchMode { get; set; } = "max";
}

public sealed class LargeImageConfig
{
    public string Mode { get; set; } = "full_quality";
    public int FastPreviewMaxSide { get; set; } = 1536;
    public float RoiCoarseThreshold { get; set; } = 0.35f;
    public int RoiPaddingPixels { get; set; } = 64;
}

public sealed record PredictionResult(
    string MaskPath,
    string OverlayPath,
    int MaskArea,
    float PredAreaRatio,
    string ModelType,
    string? ResultJsonPath = null,
    string? SummaryCsvPath = null,
    IReadOnlyDictionary<string, string>? Metadata = null);
