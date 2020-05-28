XGBoost.swift
=============

A new coming Swift interface for
[XGBoost](https://github.com/dmlc/xgboost).

The current interface is wrapping around the C API of XGBoost version
1.1.0, tries to conform to the Python API. Document see
[docs](https://ddxgz.github.io/XGBoost.swift/).

**Note**: this is not an official XGBoost project.

Installation
------------

Tested under MacOS 10.15 with `brew install xgboost`. The C header file
and library are located through `pkg-config`. Read through the following
links for configuration detail.

-   [swift-package-manager](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#requiring-system-libraries)
-   [Clang Module Map
    Language](https://clang.llvm.org/docs/Modules.html#module-map-language)
-   [Guide to
    pkg-config](https://people.freedesktop.org/~dbn/pkg-config-guide.html)

Ubuntu is tested by using [Swift's docker
image](https://swift.org/download/#docker), the latest version is
Ubuntu18.04 for now.

Usage
-----

**Still in early development, use with caution.**

```swift
let train = try DMatrix(filename: "data/agaricus.txt.train")
let test = try DMatrix(filename: "data/agaricus.txt.test")

let bst = try xgboost(data: train, numRound: 10)
let pred = bst.predict(data: test)

let cvResult = xgboostCV(data: train, numRound: 10)

// save and load model as binary
let modelBin = "bst.bin"
try bst.save(fname: modelBin)
let bstLoaded = try xgboost(data: train, numRound: 0, modelFile: modelBin)

// save and load model as json
let modelJson = "bst.json"
bst.save(fname: modelJson) 
let bstJsonLoaded = try xgboost(data: train, numRound: 0, modelFile: modelJson)

// save model config
try bst.saveConfig(fname: "config.json")
```
