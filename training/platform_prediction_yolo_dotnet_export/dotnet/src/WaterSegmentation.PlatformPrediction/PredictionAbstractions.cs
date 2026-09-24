using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WaterSegmentation.PlatformPrediction;

public sealed record PredictionRequest(
    Image<Rgb24> Image,
    string? ImagePath = null,
    IReadOnlyDictionary<string, string>? Metadata = null);

public sealed record SegmentationPrediction(
    byte[,] Mask,
    float[,]? Probability,
    string ModelType,
    IReadOnlyDictionary<string, string>? Metadata = null);

public interface IImageSegmentationPredictor : IDisposable
{
    string ModelType { get; }

    SegmentationPrediction Predict(PredictionRequest request);
}

public static class ModelRegistry
{
    private static readonly Dictionary<string, Func<string, PredictionConfig, IImageSegmentationPredictor>> Factories =
        new(StringComparer.OrdinalIgnoreCase);

    static ModelRegistry()
    {
        Register("yolo-onnx", (packageRoot, config) => new YoloOnnxSegmentationPredictor(packageRoot, config));
        Register("yolo", (packageRoot, config) => new YoloOnnxSegmentationPredictor(packageRoot, config));
        Register(
            "dual-context-onnx",
            (packageRoot, config) => new DualContextOnnxSegmentationPredictor(packageRoot, config));
    }

    public static void Register(
        string modelType,
        Func<string, PredictionConfig, IImageSegmentationPredictor> factory)
    {
        if (string.IsNullOrWhiteSpace(modelType))
        {
            throw new ArgumentException("Model type is required.", nameof(modelType));
        }

        Factories[modelType] = factory ?? throw new ArgumentNullException(nameof(factory));
    }

    public static bool IsRegistered(string modelType) => Factories.ContainsKey(modelType);

    public static IImageSegmentationPredictor Create(string packageRoot, PredictionConfig config)
    {
        if (!Factories.TryGetValue(config.ModelType, out var factory))
        {
            throw new NotSupportedException($"Unsupported modelType: {config.ModelType}");
        }

        return factory(packageRoot, config);
    }
}

public static class ImageSegmentationPredictorFactory
{
    public static IImageSegmentationPredictor Create(string packageRoot, PredictionConfig config)
        => ModelRegistry.Create(packageRoot, config);
}
