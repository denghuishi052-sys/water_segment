using WaterSegmentation.PlatformPrediction;

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage:");
    Console.Error.WriteLine("  WaterSegmentation.PlatformPrediction.Cli <packageRoot> <imagePath> <outputDir> [configPath]");
    return 2;
}

var packageRoot = Path.GetFullPath(args[0]);
var imagePath = Path.GetFullPath(args[1]);
var outputDir = Path.GetFullPath(args[2]);
var configPath = args.Length >= 4
    ? Path.GetFullPath(args[3])
    : Path.Combine(packageRoot, "configs", "dual_context.json");

var config = PredictionConfig.Load(configPath);
using var service = new PredictionService(packageRoot, config);
var result = service.PredictFile(imagePath, outputDir);

Console.WriteLine($"maskPath={result.MaskPath}");
Console.WriteLine($"overlayPath={result.OverlayPath}");
Console.WriteLine($"maskArea={result.MaskArea}");
Console.WriteLine($"predAreaRatio={result.PredAreaRatio:F6}");
Console.WriteLine($"modelType={result.ModelType}");
Console.WriteLine($"resultJsonPath={result.ResultJsonPath}");
Console.WriteLine($"summaryCsvPath={result.SummaryCsvPath}");
return 0;
