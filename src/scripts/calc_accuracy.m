conf = result.averageRun.validationPerfConfusionMatrix{end};
accuracy = trace(conf) / sum(conf,"all")
auroc = result.averageRun.validationPerf.auroc
