// #if canImport(PythonKit)
//     import PythonKit
// #else
// import Python
// #endif
// comment so that Colab does not interpret `#if ...` as a comment
// #if canImport(PythonKit)
//     import PythonKit
// #else
// import PythonKit
// #endif

import Foundation
import PythonKit
import XGBoostSwift

// let skLearnModelSelection = try Python.attemptImport("sklearn.model_selection")
// let pandas = try Python.attemptImport("pandas")

print(Python.version)

let pyxgb = try Python.attemptImport("xgboost")

print("Hello, world!")

let train = try DMatrix(fromFile: "data/agaricus.txt.train")
let test = try DMatrix(fromFile: "data/agaricus.txt.test")

let param = [
    ("objective", "binary:logistic"),
    ("max_depth", "2"),
]
let bst = try xgboost(params: param, data: train, numRound: 1)
