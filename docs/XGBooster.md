# XGBooster

``` swift
public class XGBooster
```

## Initializers

### `init(handle:)`

``` swift
internal init(handle: BoosterHandle)
```

### `init(dms:)`

``` swift
internal init(dms: inout [DMatrixHandle?])
```

## Properties

### `handle`

``` swift
var handle: BoosterHandle?
```

## Methods

### `_guardHandle()`

``` swift
private func _guardHandle() throws
```

### `save(fname:)`

Use .json as filename suffix to save model to json.
Refer to [XGBoost doc](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html)

``` swift
public func save(fname: String) throws
```

### `update(data:currentIter:)`

``` swift
public func update(data: DMatrix, currentIter: Int)
```

### `evalSet(dmHandle:evalNames:currentIter:)`

``` swift
public func evalSet(dmHandle: [DMatrix], evalNames: [String], currentIter: Int) -> String?
```

### `predict(data:outputMargin:nTreeLimit:)`

``` swift
public func predict(data: DMatrix, outputMargin: Bool = false, nTreeLimit: UInt = 0) -> [Float]
```

### `saveConfig(fname:)`

``` swift
public func saveConfig(fname: String) throws
```
