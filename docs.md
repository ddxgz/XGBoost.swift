XGBoost.swift
=============

![macOS](https://github.com/ddxgz/XGBoost.swift/workflows/macOS/badge.svg)
![Ubuntu](https://github.com/ddxgz/XGBoost.swift/workflows/Ubuntu/badge.svg)

A Swift interface for
[XGBoost](https://github.com/dmlc/xgboost).

The current interface is wrapping around the C API of XGBoost, tries to conform to the Python API. Document see
[docs](https://ddxgz.github.io/XGBoost.swift/).

If you run into any problem, please file an **issue** or even better a **pull request**.

**Note**: this is not an official XGBoost project.

- [Installation and dependency library](#installation-and-dependency-library)
  - [macOS](#macos)
  - [Ubuntu](#ubuntu)
  - [Swift package manager](#swift-package-manager)
- [Usage](#usage)
  - [DMatrix](#dmatrix)
  - [Boosting](#boosting)
  - [Callback](#callback)
  - [Custom Objective and Evaliation Function](#custom-objective-and-evaliation-function)

Installation and dependency library
------------
### macOS
You can follow [XGBoost
document](https://xgboost.readthedocs.io/en/latest/build.html) for installation
or build library. 
Develop and tested under macOS 10.15 with `brew install xgboost`. The C header file and
 library are located through `pkg-config`, it should work directly. Otherwise,
 place an pkg-config file as `/usr/local/lib/pkgconfig/xgboost.pc` with content:
 ```
prefix=/usr/local/Cellar/xgboost/1.1.0
exec_prefix=${prefix}/bin
libdir=${prefix}/lib
includedir=${prefix}/include

Name: xgboost
Description: XGBoost
Version: 1.1.0
Cflags: -I${includedir}
Libs: -L${libdir} -lxgboost
```

 Please read through the following links for more configuration detail.

-   [swift-package-manager](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#requiring-system-libraries)
-   [Clang Module Map
    Language](https://clang.llvm.org/docs/Modules.html#module-map-language)
-   [Guide to
    pkg-config](https://people.freedesktop.org/~dbn/pkg-config-guide.html)



 
### Ubuntu
Ubuntu is tested by using [Swift's docker
image](https://swift.org/download/#docker), the latest version is
Ubuntu18.04 for now.
Please follow [XGBoost
document](https://xgboost.readthedocs.io/en/latest/build.html) for installation
or build library. Or you can check the Dockerfile `Dockerfile_test_ubuntu` that
used for testing.


### Swift package manager
It evovles fastly, please constantly check the version.


```swift
.package(url: "https://github.com/ddxgz/XGBoost.swift.git", from: "0.6.0")
```

Usage
-----
You may find more cases in the test file in code repo.

**Still in early development, use with caution.**

```swift
let train = try DMatrix(fromFile: "data/agaricus.txt.train")
let test = try DMatrix(fromFile: "data/agaricus.txt.test")

let bst = try xgboost(data: train, numRound: 10)
let pred = bst.predict(data: test)

let cvResult = xgboostCV(data: train, numRound: 10)

// save and load model as binary
let modelBin = "bst.bin"
try bst.saveModel(toFile: modelBin)
let bstLoaded = try xgboost(data: train, numRound: 0, modelFile: modelBin)

// save and load model as json
let modelJson = "bst.json"
bst.saveModel(toFile: modelJson)
let bstJsonLoaded = try xgboost(data: train, numRound: 0, modelFile: modelJson)

// save model config
try bst.saveConfig(toFile: "config.json")
```

### DMatrix
```swift
// Default LibSVM text format file
let datafile = "data/agaricus.txt.train"
let train = try DMatrix(fromFile: datafile)

// Load DMatrix from csv file by set `format` parameter
let csv = "data/train.csv"
let trainCSV = try DMatrix(fromFile: csv, format: "csv")

// or by providing format URI 
let csv2 = "data/train.csv?format=csv"
let trainCSV2 = try DMatrix(fromFile: csv2, format: "csv")

// Construct from array of Floats, by setting shape, missing values will be filled
// in automatically or by setting `missing`.
let matWithNa = try DMatrix(fromArray: [Float]([0, 1, 2, 3]), shape: (10, 10))

// Data slicing by array of indexes
let trainSliced = train.slice(rows: [0, 3])!

// by range
let trainRanged = train.slice(rows: 0 ..< 10)!

// allow groups
let trainSlicedGroup = train.slice(rows: [0, 3], allowGroups: true)!

// Save DMatrix to binary file
let dmFile = "Tests/tmp/dmfile.sliced"
try trainSliced.saveBinary(toFile: dmFile)

// Get DMatrix labels
let labels = train.label

// Get weights
let weights = train.weight
// Set weights
let weightSet = [Float]([1, 3, 4])

// Get base_margins
let base_margins = train.base_margin
```

### Boosting
```swift
let train = try DMatrix(fromFile: "data/agaricus.txt.train")
let test = try DMatrix(fromFile: "data/agaricus.txt.test")

let params = [
    ("objective", "binary:logistic"),
    ("max_depth", "9"),
    ("eval_metric", "auc"),
    ("eval_metric", "aucpr"),
]
// Construct booster while boosting
let bst = try xgboost(params: param, data: train, numRound: 10)

// Set parameters by passing dictionary or name-value pair
bst.setParam(param)
bst.setParam(name: "alpha", value: "0.1")

let result = bst.predict(data: test)

// Save model to file
let modelfile = "Tests/tmp/bst.model"
try bst.saveModel(toFile: modelfile)

// Construct booster from file
let bstJsonLoaded2 = try Booster(params: params, cache: [train],
                                 modelFile: modelfileJson)

// Save config to json file
let configfile = "Tests/tmp/config.json"
try bst.saveConfig(toFile: configfile)

// Cross Validation
let cvResults = xgboostCV(params: param, data: train, numRound: 10, nFold: 5)
```

### Callback
The `SimplePrintEvalution` is a builtin simple example of callback.
You can also define a custom callback that conforms to `XGBCallback` protocol, see
more in the document of protocol.
```swift
let train = try DMatrix(fromFile: "data/agaricus.txt.train")
let test = try DMatrix(fromFile: "data/agaricus.txt.test")

let callbacks = [SimplePrintEvalution(period: 5)]
let bst = try xgboost(data: train, numRound: 10,
                      evalSet: [(train, "train"), (test, "test")],
                      callbacks: callbacks)
```

### Custom Objective and Evaliation Function
A custom objecitve function has a signature of `FuncObj`, and a custom evaluation function has a signature of `FuncEval`.

```swift
let train = try DMatrix(fromFile: "data/agaricus.txt.train")
let test = try DMatrix(fromFile: "data/agaricus.txt.test")

func dumEval(preds: [Float], dmatrix: DMatrix) -> (String, Float) {
    let labels = dmatrix.label
    let predicts = preds.map { x -> Float in
        if x > 0 { return 1.0 } else { return 0.0 }
    }
    var cnt: Float = 0
    for (label, pred) in zip(labels, predicts) {
        if label == pred {
            cnt += 1
        }
    }
    return ("dumEval", cnt / Float(labels.count))
}

func logLossObj(preds: [Float], dmatrix: DMatrix) -> ([Float], [Float]) {
    let labels = dmatrix.label
    let predicts = preds.map { x -> Float in
        Float(1.0 / (1.0 + exp(-x)))
    }

    var grad = [Float](), hess = [Float]()
    for (label, pred) in zip(labels, predicts) {
        grad.append(pred - label)
        hess.append(pred * (1.0 - pred))
    }
    return (grad, hess)
}

let callbacks = [SimplePrintEvalution(period: 5)]

let bst = try xgboost(data: train, numRound: 10,
                      evalSet: [(train, "train"), (test, "test")],
                      fnObj: logLossObj,
                      fnEval: dumEval,
                      callbacks: callbacks)
```