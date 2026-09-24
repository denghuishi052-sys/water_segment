namespace WaterSegmentation.PlatformPrediction;

public sealed record PipelineExecutionTrace(
    string Type,
    IReadOnlyList<string> EnabledSteps,
    IReadOnlyDictionary<string, string> Metadata);

public sealed class PipelineRunner
{
    private readonly PipelineConfig _config;

    public PipelineRunner(PipelineConfig config)
    {
        _config = config;
    }

    public PipelineExecutionTrace CreateTrace()
    {
        var enabledSteps = _config.Steps
            .Where(step => step.Enabled)
            .Select(step => string.IsNullOrWhiteSpace(step.Type)
                ? step.Name
                : $"{step.Name}:{step.Type}")
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .ToArray();

        var metadata = new Dictionary<string, string>
        {
            ["pipeline.type"] = _config.Type,
            ["pipeline.enabledStepCount"] = enabledSteps.Length.ToString()
        };

        if (enabledSteps.Length > 0)
        {
            metadata["pipeline.enabledSteps"] = string.Join(">", enabledSteps);
        }

        return new PipelineExecutionTrace(_config.Type, enabledSteps, metadata);
    }
}
