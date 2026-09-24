using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;

namespace WaterSegmentation.PlatformPrediction;

public sealed class PredictionService : IDisposable
{
    private readonly IImageSegmentationPredictor _predictor;
    private readonly PipelineRunner? _pipelineRunner;

    public PredictionService(string packageRoot, PredictionConfig config)
    {
        packageRoot = Path.GetFullPath(packageRoot);
        PredictionConfigValidator.ThrowIfInvalid(packageRoot, config);
        _predictor = ImageSegmentationPredictorFactory.Create(packageRoot, config);
        _pipelineRunner = new PipelineRunner(config.Pipeline);
    }

    public PredictionService(IImageSegmentationPredictor predictor)
    {
        _predictor = predictor;
    }

    public PredictionResult PredictFile(string imagePath, string outputDir, bool saveOverlay = true)
    {
        Directory.CreateDirectory(outputDir);
        var stopwatch = Stopwatch.StartNew();
        using var image = Image.Load<Rgb24>(imagePath);
        var prediction = _predictor.Predict(new PredictionRequest(image, imagePath));
        var mask = prediction.Mask;

        var stem = Path.GetFileNameWithoutExtension(imagePath);
        var maskPath = Path.Combine(outputDir, $"{stem}_pred.png");
        var overlayPath = Path.Combine(outputDir, $"{stem}_overlay.jpg");
        ImageOutput.SaveMask(mask, maskPath);
        if (saveOverlay)
        {
            ImageOutput.SaveOverlay(image, mask, overlayPath);
        }

        var area = MaskPostprocessor.CountForeground(mask);
        var ratio = area / (float)(image.Width * image.Height);
        stopwatch.Stop();

        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["image.path"] = imagePath,
            ["image.width"] = image.Width.ToString(),
            ["image.height"] = image.Height.ToString(),
            ["elapsedMilliseconds"] = stopwatch.ElapsedMilliseconds.ToString()
        };

        if (prediction.Metadata is not null)
        {
            foreach (var item in prediction.Metadata)
            {
                metadata[item.Key] = item.Value;
            }
        }

        if (_pipelineRunner is not null)
        {
            foreach (var item in _pipelineRunner.CreateTrace().Metadata)
            {
                metadata[item.Key] = item.Value;
            }
        }

        var resultJsonPath = Path.Combine(outputDir, $"{stem}_result.json");
        var summaryCsvPath = Path.Combine(outputDir, "summary.csv");
        var result = new PredictionResult(maskPath, overlayPath, area, ratio, prediction.ModelType, resultJsonPath, summaryCsvPath, metadata);
        PredictionOutput.SaveResultJson(result, resultJsonPath);
        PredictionOutput.AppendSummaryCsv(result, summaryCsvPath);
        return result;
    }

    public void Dispose() => _predictor.Dispose();
}
