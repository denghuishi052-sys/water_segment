using System.Globalization;
using System.Text;
using System.Text.Json;

namespace WaterSegmentation.PlatformPrediction;

public static class PredictionOutput
{
    public static string SaveResultJson(PredictionResult result, string path)
    {
        var payload = new
        {
            result.MaskPath,
            result.OverlayPath,
            result.MaskArea,
            result.PredAreaRatio,
            result.ModelType,
            result.Metadata
        };
        var json = JsonSerializer.Serialize(
            payload,
            new JsonSerializerOptions
            {
                WriteIndented = true
            });
        File.WriteAllText(path, json, Encoding.UTF8);
        return path;
    }

    public static string AppendSummaryCsv(PredictionResult result, string path)
    {
        var exists = File.Exists(path);
        using var writer = new StreamWriter(path, append: true, Encoding.UTF8);
        if (!exists)
        {
            writer.WriteLine("timestamp,modelType,maskArea,predAreaRatio,maskPath,overlayPath,imagePath,width,height,elapsedMilliseconds,largeImageMode,tileCount");
        }

        var metadata = result.Metadata ?? new Dictionary<string, string>();
        metadata.TryGetValue("image.path", out var imagePath);
        metadata.TryGetValue("image.width", out var width);
        metadata.TryGetValue("image.height", out var height);
        metadata.TryGetValue("elapsedMilliseconds", out var elapsed);
        metadata.TryGetValue("largeImage.mode", out var largeImageMode);
        metadata.TryGetValue("tile.count", out var tileCount);

        writer.WriteLine(string.Join(
            ",",
            Escape(DateTimeOffset.Now.ToString("O", CultureInfo.InvariantCulture)),
            Escape(result.ModelType),
            result.MaskArea.ToString(CultureInfo.InvariantCulture),
            result.PredAreaRatio.ToString("F6", CultureInfo.InvariantCulture),
            Escape(result.MaskPath),
            Escape(result.OverlayPath),
            Escape(imagePath ?? ""),
            Escape(width ?? ""),
            Escape(height ?? ""),
            Escape(elapsed ?? ""),
            Escape(largeImageMode ?? ""),
            Escape(tileCount ?? "")));

        return path;
    }

    private static string Escape(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }
}
