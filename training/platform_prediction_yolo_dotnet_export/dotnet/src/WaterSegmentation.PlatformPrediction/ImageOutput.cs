using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

namespace WaterSegmentation.PlatformPrediction;

public static class ImageOutput
{
    public static void SaveMask(byte[,] mask, string path)
    {
        var h = mask.GetLength(0);
        var w = mask.GetLength(1);
        using var image = new Image<L8>(w, h);
        for (var y = 0; y < h; y++)
        {
            var row = image.DangerousGetPixelRowMemory(y).Span;
            for (var x = 0; x < w; x++)
            {
                row[x] = new L8(mask[y, x] == 0 ? (byte)0 : (byte)255);
            }
        }
        image.Save(path);
    }

    public static void SaveOverlay(Image<Rgb24> source, byte[,] mask, string path)
    {
        using var overlay = source.Clone();
        var h = Math.Min(mask.GetLength(0), overlay.Height);
        var w = Math.Min(mask.GetLength(1), overlay.Width);
        for (var y = 0; y < h; y++)
        {
            var row = overlay.DangerousGetPixelRowMemory(y).Span;
            for (var x = 0; x < w; x++)
            {
                if (mask[y, x] == 0)
                {
                    continue;
                }

                var p = row[x];
                row[x] = new Rgb24(
                    (byte)Math.Clamp((int)(p.R * 0.55f + 40), 0, 255),
                    (byte)Math.Clamp((int)(p.G * 0.55f + 170), 0, 255),
                    (byte)Math.Clamp((int)(p.B * 0.55f + 255), 0, 255));
            }
        }
        overlay.Save(path, new JpegEncoder { Quality = 92 });
    }
}
