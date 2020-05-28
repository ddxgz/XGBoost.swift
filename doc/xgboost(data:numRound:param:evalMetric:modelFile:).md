# xgboost(data:numRound:param:evalMetric:modelFile:)

Train a booster with given parameters.

``` swift
public func xgboost(data: DMatrix, numRound: Int = 10, param: Param = [:], evalMetric: [String] = [], modelFile: String? = nil) throws -> XGBooster
```

## Parameters

  - data: - data: DMatrix
  - numRound: - numRound: Int - Number of boosting iterations.
  - param: - param: Dictionary - Booster parameters. If intend to use multiple `eval_metric`, they should be provided as the `evalMetric`.
  - evalMetric: - evalMetric: \[String\] - to pass the `eval_metric` parameter to booster.
  - modelFile: - modelFile: String - If the modelFile param is provided, it will load the model from that file.

## Returns

XGBooster
