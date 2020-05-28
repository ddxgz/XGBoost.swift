# DMatrix

``` swift
public class DMatrix
```

## Initializers

### `init(fname:silent:)`

``` swift
public init(fname: String, silent: Bool = true) throws
```

### `init(array:shape:NaValue:)`

``` swift
public init(array: [Float], shape: (row: Int, col: Int), NaValue: Float = -.infinity) throws
```

### `init(handle:)`

``` swift
internal init(handle: DMatrixHandle?)
```

## Properties

### `handle`

``` swift
var handle: DMatrixHandle?
```

### `dmHandle`

``` swift
var dmHandle: DMatrixHandle?
```

### `initialized`

``` swift
var initialized: Bool
```

### `nRow`

``` swift
var nRow: UInt64
```

### `nCol`

``` swift
var nCol: UInt64
```

### `shape`

``` swift
var shape: [UInt64]
```

### `labels`

``` swift
var labels: [Float]?
```

## Methods

### `_guardHandle()`

``` swift
private func _guardHandle() throws
```

### `save(fname:silent:)`

``` swift
public func save(fname: String, silent: Bool = true) throws
```

### `slice(rows:)`

Use rows input is an array of row index to be selected

``` swift
public func slice(rows idxSet: [Int]) -> DMatrix?
```

### `slice(rows:)`

``` swift
public func slice(rows idxSet: Range<Int>) -> DMatrix?
```
