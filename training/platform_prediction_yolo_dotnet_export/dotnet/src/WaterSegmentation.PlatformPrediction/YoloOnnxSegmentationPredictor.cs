using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WaterSegmentation.PlatformPrediction;

public sealed class YoloOnnxSegmentationPredictor : IImageSegmentationPredictor
{
    private readonly string _packageRoot;
    private readonly PredictionConfig _config;
    private readonly OnnxYoloSegmenter _primary;

    public YoloOnnxSegmentationPredictor(string packageRoot, PredictionConfig config)
    {
        _packageRoot = Path.GetFullPath(packageRoot);
        _config = config;
        _primary = new OnnxYoloSegmenter(Resolve(config.ModelPath), config);
    }

    public string ModelType => "yolo-onnx";

    public SegmentationPrediction Predict(PredictionRequest request)
    {
        var (probability, metadata) = PredictProbabilityWithMetadata(request.Image);
        var mask = MaskPostprocessor.ThresholdAndClean(
            probability,
            _config.MaskThreshold,
            _config.MinAreaRatio,
            _config.MorphClose,
            _config.SuppressLargeBorderComponents,
            _config.LargeBorderComponentMinAreaRatio,
            _config.LargeBorderComponentMarginRatio);

        return new SegmentationPrediction(mask, probability, ModelType, metadata);
    }

    public float[,] PredictProbability(Image<Rgb24> image)
        => PredictProbabilityWithMetadata(image).Probability;

    private (float[,] Probability, IReadOnlyDictionary<string, string> Metadata) PredictProbabilityWithMetadata(Image<Rgb24> image)
    {
        var isLarge = _config.Tiling.TileSize > 0 && Math.Max(image.Width, image.Height) > _config.Tiling.TileSize;
        if (!isLarge)
        {
            var probability = _primary.PredictProbability(image);
            return (probability, new Dictionary<string, string>
            {
                ["largeImage.mode"] = "single_image",
                ["tile.count"] = "1"
            });
        }

        return _config.LargeImage.Mode.ToLowerInvariant() switch
        {
            "fast_preview" => PredictFastPreview(image),
            "roi_refine" => PredictRoiRefine(image),
            _ => PredictFullQuality(image)
        };
    }

    private (float[,] Probability, IReadOnlyDictionary<string, string> Metadata) PredictFullQuality(Image<Rgb24> image)
    {
        var tileSize = _config.Tiling.TileSize;
        var overlap = _config.Tiling.OverlapPixels;
        if (tileSize <= 0 || Math.Max(image.Width, image.Height) <= tileSize)
        {
            var probability = _primary.PredictProbability(image);
            return (probability, new Dictionary<string, string>
            {
                ["largeImage.mode"] = "single_image",
                ["tile.count"] = "1"
            });
        }

        var full = new float[image.Height, image.Width];
        var tileCount = 0;
        foreach (var tile in BuildTiles(image.Width, image.Height, tileSize, overlap))
        {
            tileCount++;
            using var tileImage = image.Clone(ctx => ctx.Crop(new Rectangle(tile.X, tile.Y, tile.Width, tile.Height)));
            var tileProb = _primary.PredictProbability(tileImage);
            for (var y = 0; y < tile.Height; y++)
            {
                for (var x = 0; x < tile.Width; x++)
                {
                    var value = tileProb[y, x];
                    if (value > full[tile.Y + y, tile.X + x])
                    {
                        full[tile.Y + y, tile.X + x] = value;
                    }
                }
            }
        }
        return (full, new Dictionary<string, string>
        {
            ["largeImage.mode"] = "full_quality",
            ["tile.count"] = tileCount.ToString(),
            ["tiling.tileSize"] = tileSize.ToString(),
            ["tiling.overlapPixels"] = overlap.ToString()
        });
    }

    private (float[,] Probability, IReadOnlyDictionary<string, string> Metadata) PredictFastPreview(Image<Rgb24> image)
    {
        var maxSide = Math.Max(1, _config.LargeImage.FastPreviewMaxSide);
        var scale = maxSide / (float)Math.Max(image.Width, image.Height);
        if (scale >= 1f)
        {
            return PredictFullQuality(image);
        }

        var resizedW = Math.Max(1, (int)MathF.Round(image.Width * scale));
        var resizedH = Math.Max(1, (int)MathF.Round(image.Height * scale));
        using var resized = image.Clone(ctx => ctx.Resize(resizedW, resizedH));
        var preview = _primary.PredictProbability(resized);
        var full = ResizeProbability(preview, image.Height, image.Width);
        return (full, new Dictionary<string, string>
        {
            ["largeImage.mode"] = "fast_preview",
            ["largeImage.fastPreviewMaxSide"] = maxSide.ToString(),
            ["largeImage.previewWidth"] = resizedW.ToString(),
            ["largeImage.previewHeight"] = resizedH.ToString(),
            ["tile.count"] = "1"
        });
    }

    private (float[,] Probability, IReadOnlyDictionary<string, string> Metadata) PredictRoiRefine(Image<Rgb24> image)
    {
        var (coarse, coarseMetadata) = PredictFastPreview(image);
        var bbox = FindBoundingBox(coarse, _config.LargeImage.RoiCoarseThreshold);
        if (bbox is null)
        {
            var empty = new float[image.Height, image.Width];
            var noRoiMetadata = new Dictionary<string, string>(coarseMetadata)
            {
                ["largeImage.mode"] = "roi_refine",
                ["largeImage.roi.found"] = "false",
                ["tile.count"] = "0"
            };
            return (empty, noRoiMetadata);
        }

        var roi = Expand(bbox.Value, image.Width, image.Height, _config.LargeImage.RoiPaddingPixels);
        using var roiImage = image.Clone(ctx => ctx.Crop(roi));
        var (roiProb, roiMetadata) = PredictFullQuality(roiImage);
        var full = new float[image.Height, image.Width];
        for (var y = 0; y < roi.Height; y++)
        {
            for (var x = 0; x < roi.Width; x++)
            {
                full[roi.Y + y, roi.X + x] = roiProb[y, x];
            }
        }

        roiMetadata.TryGetValue("tile.count", out var tileCount);
        return (full, new Dictionary<string, string>
        {
            ["largeImage.mode"] = "roi_refine",
            ["largeImage.roi.found"] = "true",
            ["largeImage.roi.x"] = roi.X.ToString(),
            ["largeImage.roi.y"] = roi.Y.ToString(),
            ["largeImage.roi.width"] = roi.Width.ToString(),
            ["largeImage.roi.height"] = roi.Height.ToString(),
            ["largeImage.roiCoarseThreshold"] = _config.LargeImage.RoiCoarseThreshold.ToString(),
            ["tile.count"] = tileCount ?? "1"
        });
    }

    private static IEnumerable<Rectangle> BuildTiles(int width, int height, int tileSize, int overlap)
    {
        var step = Math.Max(1, tileSize - overlap);
        var xs = Starts(width, tileSize, step);
        var ys = Starts(height, tileSize, step);
        foreach (var y in ys)
        {
            foreach (var x in xs)
            {
                yield return new Rectangle(x, y, Math.Min(tileSize, width - x), Math.Min(tileSize, height - y));
            }
        }
    }

    private static IReadOnlyList<int> Starts(int length, int tileSize, int step)
    {
        if (length <= tileSize)
        {
            return [0];
        }

        var values = new List<int>();
        for (var start = 0; start < length; start += step)
        {
            var adjusted = Math.Min(start, length - tileSize);
            if (values.Count == 0 || values[^1] != adjusted)
            {
                values.Add(adjusted);
            }
            if (adjusted + tileSize >= length)
            {
                break;
            }
        }
        return values;
    }

    private string Resolve(string path) => Path.IsPathRooted(path) ? path : Path.Combine(_packageRoot, path);

    public void Dispose() => _primary.Dispose();

    private static float[,] ResizeProbability(float[,] source, int targetH, int targetW)
    {
        var sourceH = source.GetLength(0);
        var sourceW = source.GetLength(1);
        var target = new float[targetH, targetW];
        for (var y = 0; y < targetH; y++)
        {
            var srcY = (y + 0.5f) * sourceH / targetH - 0.5f;
            for (var x = 0; x < targetW; x++)
            {
                var srcX = (x + 0.5f) * sourceW / targetW - 0.5f;
                target[y, x] = Bilinear(source, srcX, srcY);
            }
        }
        return target;
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

    private static Rectangle? FindBoundingBox(float[,] probability, float threshold)
    {
        var h = probability.GetLength(0);
        var w = probability.GetLength(1);
        var minX = w;
        var minY = h;
        var maxX = -1;
        var maxY = -1;
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (probability[y, x] < threshold)
                {
                    continue;
                }

                minX = Math.Min(minX, x);
                minY = Math.Min(minY, y);
                maxX = Math.Max(maxX, x);
                maxY = Math.Max(maxY, y);
            }
        }

        return maxX < minX || maxY < minY
            ? null
            : new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private static Rectangle Expand(Rectangle rect, int width, int height, int padding)
    {
        var x = Math.Max(0, rect.X - padding);
        var y = Math.Max(0, rect.Y - padding);
        var right = Math.Min(width, rect.Right + padding);
        var bottom = Math.Min(height, rect.Bottom + padding);
        return new Rectangle(x, y, Math.Max(1, right - x), Math.Max(1, bottom - y));
    }
}
