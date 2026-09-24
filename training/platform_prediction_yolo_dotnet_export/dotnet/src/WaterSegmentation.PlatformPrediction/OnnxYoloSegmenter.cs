using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WaterSegmentation.PlatformPrediction;

public sealed class OnnxYoloSegmenter : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly PredictionConfig _config;

    public OnnxYoloSegmenter(string modelPath, PredictionConfig config)
    {
        _config = config;
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public float[,] PredictProbability(Image<Rgb24> image)
    {
        var (tensor, meta) = Preprocessor.LetterboxToTensor(image, _config.ImageSize);
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run([input]);
        var outputs = results.ToArray();
        if (outputs.Length < 2)
        {
            throw new InvalidOperationException("YOLOv8-seg ONNX must return output0 and output1.");
        }

        var output0 = outputs[0].AsTensor<float>();
        var output1 = outputs[1].AsTensor<float>();
        return YoloV8SegPostprocessor.DecodeProbability(
            output0,
            output1,
            meta,
            image.Height,
            image.Width,
            _config);
    }

    public void Dispose() => _session.Dispose();
}
