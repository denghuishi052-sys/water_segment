using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WaterSegmentation.PlatformPrediction;

public sealed class DualContextOnnxSegmentationPredictor : IImageSegmentationPredictor
{
    private static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];

    private readonly InferenceSession _session;
    private readonly PredictionConfig _config;
    private readonly int _localSize;
    private readonly int _globalSize;
    private readonly int _overlap;
    private readonly float[,] _tileWeight;

    public DualContextOnnxSegmentationPredictor(string packageRoot, PredictionConfig config)
    {
        _config = config;
        _localSize = config.ImageSize;
        _globalSize = config.GetModelOption("globalSize", 512);
        _overlap = config.Tiling.OverlapPixels;
        var modelPath = Path.IsPathRooted(config.ModelPath)
            ? config.ModelPath
            : Path.Combine(packageRoot, config.ModelPath);
        _session = new InferenceSession(modelPath);
        ValidateModelContract();
        _tileWeight = CreateHannWeight(_localSize);
    }

    public string ModelType => "dual-context-onnx";

    public SegmentationPrediction Predict(PredictionRequest request)
    {
        var image = request.Image;
        var globalRgb = CreateGlobalRgbTensor(image);
        var probabilitySum = new float[image.Height, image.Width];
        var weightSum = new float[image.Height, image.Width];
        var qualitySum = 0f;
        var tileCount = 0;

        foreach (var tile in BuildTiles(image.Width, image.Height, _localSize, _overlap))
        {
            tileCount++;
            var local = CreateLocalTensor(image, tile);
            var context = CreateGlobalContext(globalRgb, tile, image.Width, image.Height);
            using var results = _session.Run(
            [
                NamedOnnxValue.CreateFromTensor("local_image", local),
                NamedOnnxValue.CreateFromTensor("global_context", context)
            ]);
            var logits = results.Single(result => result.Name == "water_logits").AsTensor<float>();
            var quality = results.Single(result => result.Name == "quality_logits").AsTensor<float>();
            qualitySum += Sigmoid(quality[0, 0]);

            for (var y = 0; y < tile.Height; y++)
            {
                for (var x = 0; x < tile.Width; x++)
                {
                    var weight = _tileWeight[y, x];
                    probabilitySum[tile.Y + y, tile.X + x] += Sigmoid(logits[0, 0, y, x]) * weight;
                    weightSum[tile.Y + y, tile.X + x] += weight;
                }
            }
        }

        var probability = new float[image.Height, image.Width];
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                probability[y, x] = probabilitySum[y, x] / Math.Max(weightSum[y, x], 1e-6f);
            }
        }

        var mask = MaskPostprocessor.ThresholdAndClean(
            probability,
            _config.MaskThreshold,
            _config.MinAreaRatio,
            _config.MorphClose,
            _config.SuppressLargeBorderComponents,
            _config.LargeBorderComponentMinAreaRatio,
            _config.LargeBorderComponentMarginRatio);
        return new SegmentationPrediction(
            mask,
            probability,
            ModelType,
            new Dictionary<string, string>
            {
                ["largeImage.mode"] = "dual_context_full_quality",
                ["tile.count"] = tileCount.ToString(),
                ["tiling.tileSize"] = _localSize.ToString(),
                ["tiling.overlapPixels"] = _overlap.ToString(),
                ["globalContext.size"] = _globalSize.ToString(),
                ["quality.mean"] = (qualitySum / Math.Max(tileCount, 1)).ToString("F6")
            });
    }

    public void Dispose() => _session.Dispose();

    private void ValidateModelContract()
    {
        ValidateShape("local_image", [1, 3, _localSize, _localSize]);
        ValidateShape("global_context", [1, 4, _globalSize, _globalSize]);
        if (!_session.OutputMetadata.ContainsKey("water_logits") ||
            !_session.OutputMetadata.ContainsKey("quality_logits"))
        {
            throw new InvalidOperationException(
                "Dual-context ONNX must return water_logits and quality_logits.");
        }
    }

    private void ValidateShape(string name, int[] expected)
    {
        if (!_session.InputMetadata.TryGetValue(name, out var metadata) ||
            !metadata.Dimensions.SequenceEqual(expected))
        {
            var actual = metadata is null ? "missing" : string.Join("x", metadata.Dimensions);
            throw new InvalidOperationException(
                $"ONNX input {name} must be {string.Join("x", expected)}, actual: {actual}.");
        }
    }

    private DenseTensor<float> CreateGlobalRgbTensor(Image<Rgb24> image)
    {
        using var resized = image.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new Size(_globalSize, _globalSize),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Box
        }));
        var tensor = new DenseTensor<float>([3, _globalSize, _globalSize]);
        for (var y = 0; y < _globalSize; y++)
        {
            var row = resized.DangerousGetPixelRowMemory(y).Span;
            for (var x = 0; x < _globalSize; x++)
            {
                WriteNormalizedRgb(tensor, row[x], y, x);
            }
        }
        return tensor;
    }

    private DenseTensor<float> CreateLocalTensor(Image<Rgb24> image, Tile tile)
    {
        var tensor = new DenseTensor<float>([1, 3, _localSize, _localSize]);
        for (var y = 0; y < _localSize; y++)
        {
            var sourceY = tile.Y + Reflect101(y, tile.Height);
            var row = image.DangerousGetPixelRowMemory(sourceY).Span;
            for (var x = 0; x < _localSize; x++)
            {
                var sourceX = tile.X + Reflect101(x, tile.Width);
                var pixel = row[sourceX];
                tensor[0, 0, y, x] = Normalize(pixel.R, 0);
                tensor[0, 1, y, x] = Normalize(pixel.G, 1);
                tensor[0, 2, y, x] = Normalize(pixel.B, 2);
            }
        }
        return tensor;
    }

    private DenseTensor<float> CreateGlobalContext(
        DenseTensor<float> globalRgb,
        Tile tile,
        int imageWidth,
        int imageHeight)
    {
        var tensor = new DenseTensor<float>([1, 4, _globalSize, _globalSize]);
        for (var channel = 0; channel < 3; channel++)
        {
            for (var y = 0; y < _globalSize; y++)
            {
                for (var x = 0; x < _globalSize; x++)
                {
                    tensor[0, channel, y, x] = globalRgb[channel, y, x];
                }
            }
        }

        var x0 = Math.Clamp((int)MathF.Round(tile.X * _globalSize / (float)imageWidth), 0, _globalSize);
        var y0 = Math.Clamp((int)MathF.Round(tile.Y * _globalSize / (float)imageHeight), 0, _globalSize);
        var x1 = Math.Clamp((int)MathF.Round(tile.Right * _globalSize / (float)imageWidth), 0, _globalSize);
        var y1 = Math.Clamp((int)MathF.Round(tile.Bottom * _globalSize / (float)imageHeight), 0, _globalSize);
        for (var y = y0; y < y1; y++)
        {
            for (var x = x0; x < x1; x++)
            {
                tensor[0, 3, y, x] = 1f;
            }
        }
        return tensor;
    }

    private static void WriteNormalizedRgb(DenseTensor<float> tensor, Rgb24 pixel, int y, int x)
    {
        tensor[0, y, x] = Normalize(pixel.R, 0);
        tensor[1, y, x] = Normalize(pixel.G, 1);
        tensor[2, y, x] = Normalize(pixel.B, 2);
    }

    private static float Normalize(byte value, int channel)
        => (value / 255f - ImageNetMean[channel]) / ImageNetStd[channel];

    private static int Reflect101(int index, int length)
    {
        if (index < length || length <= 1)
        {
            return Math.Min(index, length - 1);
        }
        var period = 2 * length - 2;
        var value = index % period;
        return value < length ? value : period - value;
    }

    private static float Sigmoid(float value)
        => 1f / (1f + MathF.Exp(-Math.Clamp(value, -30f, 30f)));

    private static float[,] CreateHannWeight(int size)
    {
        var axis = new float[size];
        for (var index = 0; index < size; index++)
        {
            axis[index] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * index / Math.Max(size - 1, 1));
        }
        var weight = new float[size, size];
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                weight[y, x] = 0.08f + 0.92f * axis[y] * axis[x];
            }
        }
        return weight;
    }

    private static IEnumerable<Tile> BuildTiles(int width, int height, int size, int overlap)
    {
        var stride = size - overlap;
        if (stride <= 0)
        {
            throw new InvalidOperationException("tiling overlap must be smaller than imageSize.");
        }
        foreach (var y in Starts(height, size, stride))
        {
            foreach (var x in Starts(width, size, stride))
            {
                yield return new Tile(x, y, Math.Min(size, width - x), Math.Min(size, height - y));
            }
        }
    }

    private static IReadOnlyList<int> Starts(int length, int size, int stride)
    {
        if (length <= size)
        {
            return [0];
        }
        var values = Enumerable.Range(0, (length - size) / stride + 1)
            .Select(index => index * stride)
            .ToList();
        var final = length - size;
        if (values[^1] != final)
        {
            values.Add(final);
        }
        return values;
    }

    private readonly record struct Tile(int X, int Y, int Width, int Height)
    {
        public int Right => X + Width;
        public int Bottom => Y + Height;
    }
}
