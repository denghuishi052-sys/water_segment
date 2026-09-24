namespace WaterSegmentation.PlatformPrediction;

public static class MaskPostprocessor
{
    public static byte[,] ThresholdAndClean(
        float[,] probability,
        float threshold,
        float minAreaRatio,
        bool morphClose,
        bool suppressLargeBorderComponents = false,
        float largeBorderComponentMinAreaRatio = 0.15f,
        float largeBorderComponentMarginRatio = 0.03f)
    {
        var h = probability.GetLength(0);
        var w = probability.GetLength(1);
        var mask = new byte[h, w];
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                mask[y, x] = probability[y, x] >= threshold ? (byte)1 : (byte)0;
            }
        }

        if (suppressLargeBorderComponents)
        {
            RemoveLargeBorderComponents(
                mask,
                Math.Max(1, (int)MathF.Round(h * w * largeBorderComponentMinAreaRatio)),
                Math.Max(1, (int)MathF.Round(Math.Min(h, w) * largeBorderComponentMarginRatio)));
        }
        RemoveSmallComponents(mask, Math.Max(1, (int)MathF.Round(h * w * minAreaRatio)));
        if (morphClose)
        {
            mask = Erode(Dilate(mask));
        }
        return mask;
    }

    public static int CountForeground(byte[,] mask)
    {
        var count = 0;
        foreach (var value in mask)
        {
            if (value != 0)
            {
                count++;
            }
        }
        return count;
    }

    private static void RemoveSmallComponents(byte[,] mask, int minArea)
        => RemoveComponents(mask, (pixels, touchesBorder) => pixels.Count < minArea);

    private static void RemoveLargeBorderComponents(byte[,] mask, int minArea, int borderMargin)
        => RemoveComponents(
            mask,
            (pixels, touchesBorder, minX, minY, maxX, maxY, width, height) =>
            {
                var nearBorder =
                    minX <= borderMargin ||
                    minY <= borderMargin ||
                    maxX >= width - 1 - borderMargin ||
                    maxY >= height - 1 - borderMargin;
                return nearBorder && pixels.Count >= minArea;
            });

    private static void RemoveComponents(byte[,] mask, Func<List<(int X, int Y)>, bool, bool> shouldRemove)
        => RemoveComponents(
            mask,
            (pixels, touchesBorder, minX, minY, maxX, maxY, width, height) => shouldRemove(pixels, touchesBorder));

    private static void RemoveComponents(
        byte[,] mask,
        Func<List<(int X, int Y)>, bool, int, int, int, int, int, int, bool> shouldRemove)
    {
        var h = mask.GetLength(0);
        var w = mask.GetLength(1);
        var visited = new bool[h, w];
        var queue = new Queue<(int X, int Y)>();
        var pixels = new List<(int X, int Y)>();

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (mask[y, x] == 0 || visited[y, x])
                {
                    continue;
                }

                pixels.Clear();
                var touchesBorder = false;
                var minX = x;
                var maxX = x;
                var minY = y;
                var maxY = y;
                visited[y, x] = true;
                queue.Enqueue((x, y));
                while (queue.Count > 0)
                {
                    var p = queue.Dequeue();
                    pixels.Add(p);
                    touchesBorder |= p.X == 0 || p.Y == 0 || p.X == w - 1 || p.Y == h - 1;
                    minX = Math.Min(minX, p.X);
                    maxX = Math.Max(maxX, p.X);
                    minY = Math.Min(minY, p.Y);
                    maxY = Math.Max(maxY, p.Y);
                    foreach (var n in Neighbors4(p.X, p.Y, w, h))
                    {
                        if (mask[n.Y, n.X] != 0 && !visited[n.Y, n.X])
                        {
                            visited[n.Y, n.X] = true;
                            queue.Enqueue(n);
                        }
                    }
                }

                if (shouldRemove(pixels, touchesBorder, minX, minY, maxX, maxY, w, h))
                {
                    foreach (var p in pixels)
                    {
                        mask[p.Y, p.X] = 0;
                    }
                }
            }
        }
    }

    private static byte[,] Dilate(byte[,] source)
    {
        var h = source.GetLength(0);
        var w = source.GetLength(1);
        var dest = new byte[h, w];
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (source[y, x] != 0 ||
                    (x > 0 && source[y, x - 1] != 0) ||
                    (x + 1 < w && source[y, x + 1] != 0) ||
                    (y > 0 && source[y - 1, x] != 0) ||
                    (y + 1 < h && source[y + 1, x] != 0))
                {
                    dest[y, x] = 1;
                }
            }
        }
        return dest;
    }

    private static byte[,] Erode(byte[,] source)
    {
        var h = source.GetLength(0);
        var w = source.GetLength(1);
        var dest = new byte[h, w];
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (source[y, x] != 0 &&
                    x > 0 && source[y, x - 1] != 0 &&
                    x + 1 < w && source[y, x + 1] != 0 &&
                    y > 0 && source[y - 1, x] != 0 &&
                    y + 1 < h && source[y + 1, x] != 0)
                {
                    dest[y, x] = 1;
                }
            }
        }
        return dest;
    }

    private static IEnumerable<(int X, int Y)> Neighbors4(int x, int y, int w, int h)
    {
        if (x > 0) yield return (x - 1, y);
        if (x + 1 < w) yield return (x + 1, y);
        if (y > 0) yield return (x, y - 1);
        if (y + 1 < h) yield return (x, y + 1);
    }
}
