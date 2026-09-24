using Microsoft.ML.OnnxRuntime.Tensors;

namespace WaterSegmentation.PlatformPrediction;

internal readonly record struct Detection(float X1, float Y1, float X2, float Y2, float Score, float[] Coefficients);

public static class YoloV8SegPostprocessor
{
    public static float[,] DecodeProbability(
        Tensor<float> output0,
        Tensor<float> output1,
        LetterboxMeta meta,
        int originalHeight,
        int originalWidth,
        PredictionConfig config)
    {
        if (output0.Dimensions.Length != 3 || output1.Dimensions.Length != 4)
        {
            throw new InvalidOperationException("Unexpected YOLO output rank.");
        }

        var anchors = output0.Dimensions[2];
        var nc = config.ClassCount;
        var nm = config.MaskPrototypeCount;
        var detections = new List<Detection>();

        for (var a = 0; a < anchors; a++)
        {
            var score = output0[0, 4, a];
            if (nc > 1)
            {
                score = float.MinValue;
                for (var c = 1; c < nc; c++)
                {
                    score = MathF.Max(score, output0[0, 4 + c, a]);
                }
            }

            if (score <= config.ConfidenceThreshold)
            {
                continue;
            }

            var cx = output0[0, 0, a];
            var cy = output0[0, 1, a];
            var w = output0[0, 2, a];
            var h = output0[0, 3, a];
            var coefficients = new float[nm];
            for (var i = 0; i < nm; i++)
            {
                coefficients[i] = output0[0, 4 + nc + i, a];
            }

            detections.Add(new Detection(
                cx - w / 2f,
                cy - h / 2f,
                cx + w / 2f,
                cy + h / 2f,
                score,
                coefficients));
        }

        if (detections.Count == 0)
        {
            return new float[originalHeight, originalWidth];
        }

        var keep = NonMaximumSuppression(detections, config.IouThreshold)
            .Take(config.MaxDetections)
            .Select(i => detections[i])
            .ToArray();

        var protoNm = output1.Dimensions[1];
        var protoH = output1.Dimensions[2];
        var protoW = output1.Dimensions[3];
        if (protoNm != nm)
        {
            throw new InvalidOperationException($"Prototype count mismatch: config={nm}, output={protoNm}.");
        }

        var canvas = new float[originalHeight, originalWidth];
        foreach (var det in keep)
        {
            var lowMask = BuildLowMask(output1, det.Coefficients, nm, protoH, protoW);
            PlaceMask(canvas, lowMask, protoH, protoW, det, meta, config.MaskBoxExpandRatio);
        }

        return canvas;
    }

    private static float[,] BuildLowMask(
        Tensor<float> proto,
        float[] coefficients,
        int nm,
        int protoH,
        int protoW)
    {
        var low = new float[protoH, protoW];
        for (var y = 0; y < protoH; y++)
        {
            for (var x = 0; x < protoW; x++)
            {
                var sum = 0f;
                for (var i = 0; i < nm; i++)
                {
                    sum += coefficients[i] * proto[0, i, y, x];
                }
                low[y, x] = Sigmoid(sum);
            }
        }
        return low;
    }

    private static void PlaceMask(
        float[,] canvas,
        float[,] lowMask,
        int protoH,
        int protoW,
        Detection det,
        LetterboxMeta meta,
        float expandRatio)
    {
        var x1 = det.X1;
        var y1 = det.Y1;
        var x2 = det.X2;
        var y2 = det.Y2;
        if (expandRatio > 0)
        {
            var w = x2 - x1;
            var h = y2 - y1;
            var dx = w * expandRatio * 0.5f;
            var dy = h * expandRatio * 0.5f;
            x1 = Math.Clamp(x1 - dx, 0, meta.ImageSize);
            y1 = Math.Clamp(y1 - dy, 0, meta.ImageSize);
            x2 = Math.Clamp(x2 + dx, 0, meta.ImageSize);
            y2 = Math.Clamp(y2 + dy, 0, meta.ImageSize);
        }

        var protoScale = protoH / (float)meta.ImageSize;
        var x1p = Math.Clamp((int)MathF.Floor(x1 * protoScale), 0, protoW - 1);
        var y1p = Math.Clamp((int)MathF.Floor(y1 * protoScale), 0, protoH - 1);
        var x2p = Math.Clamp((int)MathF.Ceiling(x2 * protoScale), x1p + 1, protoW);
        var y2p = Math.Clamp((int)MathF.Ceiling(y2 * protoScale), y1p + 1, protoH);

        var x1o = Math.Clamp((x1 - meta.PadW) / meta.Ratio, 0, canvas.GetLength(1));
        var y1o = Math.Clamp((y1 - meta.PadH) / meta.Ratio, 0, canvas.GetLength(0));
        var x2o = Math.Clamp((x2 - meta.PadW) / meta.Ratio, 0, canvas.GetLength(1));
        var y2o = Math.Clamp((y2 - meta.PadH) / meta.Ratio, 0, canvas.GetLength(0));
        var xStart = Math.Max(0, (int)MathF.Floor(x1o));
        var yStart = Math.Max(0, (int)MathF.Floor(y1o));
        var xEnd = Math.Min(canvas.GetLength(1), (int)MathF.Ceiling(x2o));
        var yEnd = Math.Min(canvas.GetLength(0), (int)MathF.Ceiling(y2o));

        if (xEnd <= xStart || yEnd <= yStart)
        {
            return;
        }

        for (var yy = yStart; yy < yEnd; yy++)
        {
            var letterboxY = yy * meta.Ratio + meta.PadH;
            var srcY = letterboxY * protoScale;
            if (srcY < y1p || srcY >= y2p)
            {
                continue;
            }

            for (var xx = xStart; xx < xEnd; xx++)
            {
                var letterboxX = xx * meta.Ratio + meta.PadW;
                var srcX = letterboxX * protoScale;
                if (srcX < x1p || srcX >= x2p)
                {
                    continue;
                }

                var value = Bilinear(lowMask, srcX, srcY);
                if (value > canvas[yy, xx])
                {
                    canvas[yy, xx] = value;
                }
            }
        }
    }

    private static float Bilinear(float[,] source, float x, float y)
    {
        var h = source.GetLength(0);
        var w = source.GetLength(1);
        x = Math.Clamp(x, 0, w - 1);
        y = Math.Clamp(y, 0, h - 1);
        var x0 = (int)MathF.Floor(x);
        var y0 = (int)MathF.Floor(y);
        var x1 = Math.Min(w - 1, x0 + 1);
        var y1 = Math.Min(h - 1, y0 + 1);
        var dx = x - x0;
        var dy = y - y0;
        var top = source[y0, x0] * (1 - dx) + source[y0, x1] * dx;
        var bottom = source[y1, x0] * (1 - dx) + source[y1, x1] * dx;
        return top * (1 - dy) + bottom * dy;
    }

    private static IReadOnlyList<int> NonMaximumSuppression(IReadOnlyList<Detection> detections, float threshold)
    {
        var order = Enumerable.Range(0, detections.Count)
            .OrderByDescending(i => detections[i].Score)
            .ToList();
        var keep = new List<int>();
        while (order.Count > 0)
        {
            var current = order[0];
            keep.Add(current);
            order.RemoveAt(0);
            order = order.Where(i => Iou(detections[current], detections[i]) <= threshold).ToList();
        }
        return keep;
    }

    private static float Iou(Detection a, Detection b)
    {
        var x1 = MathF.Max(a.X1, b.X1);
        var y1 = MathF.Max(a.Y1, b.Y1);
        var x2 = MathF.Min(a.X2, b.X2);
        var y2 = MathF.Min(a.Y2, b.Y2);
        var inter = MathF.Max(0, x2 - x1) * MathF.Max(0, y2 - y1);
        var areaA = MathF.Max(0, a.X2 - a.X1) * MathF.Max(0, a.Y2 - a.Y1);
        var areaB = MathF.Max(0, b.X2 - b.X1) * MathF.Max(0, b.Y2 - b.Y1);
        return inter / (areaA + areaB - inter + 1e-7f);
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0)
        {
            return 1f / (1f + MathF.Exp(-x));
        }
        var ex = MathF.Exp(x);
        return ex / (1f + ex);
    }
}
