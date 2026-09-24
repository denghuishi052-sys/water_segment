using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WaterSegmentation.PlatformPrediction;

public readonly record struct LetterboxMeta(
    float Ratio,
    int PadW,
    int PadH,
    int NewW,
    int NewH,
    int ImageSize);

public static class Preprocessor
{
    private static readonly Rgb24 PadColor = new(114, 114, 114);

    public static (DenseTensor<float> Tensor, LetterboxMeta Meta) LetterboxToTensor(
        Image<Rgb24> source,
        int imageSize)
    {
        var (canvas, meta) = Letterbox(source, imageSize);
        try
        {
            var tensor = new DenseTensor<float>([1, 3, imageSize, imageSize]);
            for (var y = 0; y < imageSize; y++)
            {
                var row = canvas.DangerousGetPixelRowMemory(y).Span;
                for (var x = 0; x < imageSize; x++)
                {
                    var p = row[x];
                    tensor[0, 0, y, x] = p.R / 255f;
                    tensor[0, 1, y, x] = p.G / 255f;
                    tensor[0, 2, y, x] = p.B / 255f;
                }
            }
            return (tensor, meta);
        }
        finally
        {
            canvas.Dispose();
        }
    }

    public static (Image<Rgb24> Canvas, LetterboxMeta Meta) Letterbox(
        Image<Rgb24> source,
        int imageSize)
    {
        var ratio = imageSize / (float)Math.Max(source.Width, source.Height);
        var newW = Math.Max(1, (int)MathF.Round(source.Width * ratio));
        var newH = Math.Max(1, (int)MathF.Round(source.Height * ratio));
        var padW = (imageSize - newW) / 2;
        var padH = (imageSize - newH) / 2;

        var resized = source.Clone(ctx => ctx.Resize(newW, newH));
        var canvas = new Image<Rgb24>(imageSize, imageSize, PadColor);
        canvas.Mutate(ctx => ctx.DrawImage(resized, new Point(padW, padH), 1f));
        resized.Dispose();

        return (canvas, new LetterboxMeta(ratio, padW, padH, newW, newH, imageSize));
    }
}
