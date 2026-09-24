namespace WaterSegmentation.PlatformPrediction;

public sealed record PredictionConfigValidationResult(IReadOnlyList<string> Errors)
{
    public bool IsValid => Errors.Count == 0;
}

public static class PredictionConfigValidator
{
    public static PredictionConfigValidationResult Validate(string packageRoot, PredictionConfig config)
    {
        var errors = new List<string>();

        if (string.IsNullOrWhiteSpace(config.ModelType))
        {
            errors.Add("modelType is required.");
        }
        else if (!ModelRegistry.IsRegistered(config.ModelType))
        {
            errors.Add($"modelType is not registered: {config.ModelType}");
        }

        if (string.IsNullOrWhiteSpace(config.ModelPath))
        {
            errors.Add("modelPath is required.");
        }
        else
        {
            var modelPath = Path.IsPathRooted(config.ModelPath)
                ? config.ModelPath
                : Path.Combine(packageRoot, config.ModelPath);
            if (!File.Exists(modelPath))
            {
                errors.Add($"modelPath does not exist: {modelPath}");
            }
        }

        AddRangeError(errors, nameof(config.ConfidenceThreshold), config.ConfidenceThreshold, 0f, 1f);
        AddRangeError(errors, nameof(config.IouThreshold), config.IouThreshold, 0f, 1f);
        AddRangeError(errors, nameof(config.MaskThreshold), config.MaskThreshold, 0f, 1f);
        AddRangeError(errors, nameof(config.MinAreaRatio), config.MinAreaRatio, 0f, 1f);
        AddRangeError(errors, nameof(config.LargeBorderComponentMinAreaRatio), config.LargeBorderComponentMinAreaRatio, 0f, 1f);
        AddRangeError(errors, nameof(config.LargeBorderComponentMarginRatio), config.LargeBorderComponentMarginRatio, 0f, 1f);
        AddRangeError(errors, "largeImage.roiCoarseThreshold", config.LargeImage.RoiCoarseThreshold, 0f, 1f);

        if (config.ImageSize <= 0)
        {
            errors.Add("imageSize must be greater than 0.");
        }

        if (config.MaxDetections <= 0)
        {
            errors.Add("maxDetections must be greater than 0.");
        }

        if (config.ClassCount <= 0)
        {
            errors.Add("classCount must be greater than 0.");
        }

        if (config.MaskPrototypeCount <= 0)
        {
            errors.Add("maskPrototypeCount must be greater than 0.");
        }

        if (config.Tiling.TileSize < 0)
        {
            errors.Add("tiling.tileSize must be greater than or equal to 0.");
        }

        if (config.Tiling.OverlapPixels < 0)
        {
            errors.Add("tiling.overlapPixels must be greater than or equal to 0.");
        }

        if (config.Tiling.TileSize > 0 && config.Tiling.OverlapPixels >= config.Tiling.TileSize)
        {
            errors.Add("tiling.overlapPixels must be less than tiling.tileSize.");
        }

        var largeImageModes = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "full_quality",
            "fast_preview",
            "roi_refine"
        };
        if (!largeImageModes.Contains(config.LargeImage.Mode))
        {
            errors.Add($"largeImage.mode is unsupported: {config.LargeImage.Mode}");
        }

        if (config.LargeImage.FastPreviewMaxSide <= 0)
        {
            errors.Add("largeImage.fastPreviewMaxSide must be greater than 0.");
        }

        if (config.LargeImage.RoiPaddingPixels < 0)
        {
            errors.Add("largeImage.roiPaddingPixels must be greater than or equal to 0.");
        }

        var duplicateStepNames = config.Pipeline.Steps
            .Where(step => !string.IsNullOrWhiteSpace(step.Name))
            .GroupBy(step => step.Name, StringComparer.OrdinalIgnoreCase)
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToArray();
        foreach (var name in duplicateStepNames)
        {
            errors.Add($"pipeline step name is duplicated: {name}");
        }

        return new PredictionConfigValidationResult(errors);
    }

    public static void ThrowIfInvalid(string packageRoot, PredictionConfig config)
    {
        var result = Validate(packageRoot, config);
        if (!result.IsValid)
        {
            throw new InvalidOperationException("Invalid prediction config: " + string.Join("; ", result.Errors));
        }
    }

    private static void AddRangeError(List<string> errors, string name, float value, float min, float max)
    {
        if (value < min || value > max)
        {
            errors.Add($"{name} must be between {min} and {max}.");
        }
    }
}
