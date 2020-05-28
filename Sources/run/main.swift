import Cxgb
import Foundation
import XGBoostSwift

print("Hello, world!")

/// use macro to safely call fn
// func safeCall(fn)

// var dTrain: DMatrixHandle?
// // var dTest: DMatrixHandle?

// var a = XGDMatrixCreateFromFile("data/agaricus.txt.tain", 1, &dTrain)
// // XGDMatrixCreateFromFile("data/agaricus.txt.test", 0, &dTest)

// print("err type: \(type(of: err))")
// let errmsg = [String](buf)
// print("err buf: \(buf)")

var dTrain = DMatrixFromFile(name: "data/agaricus.txt.train")
var dTest = DMatrixFromFile(name: "data/agaricus.txt.test")
// print(trainMat)

print(lastError())
// DMatrixFree(&dTrain!)
// let err = XGBGetLastError()
// // let buf = UnsafeBufferPointer(start: err!, count: 2)
// print("err: \(err!.pointee)")
// print("err str: \(String(cString: err!))")

let nRow = DMatrixNumRow(dTrain!)
print(nRow!)
let nCol = DMatrixNumCol(dTrain!)
print(nCol!)

let nRowTest = DMatrixNumRow(dTest!)
print("dtest rows: \(nRowTest!)")
// var major: UnsafeMutablePointer<Int32>?
// var minor: UnsafeMutablePointer<Int32>?
// var patch: UnsafeMutablePointer<Int32>?
// // var major: Int32?
// // var minor: Int32?
// // var patch: Int32?

// XGBoostVersion(major, minor, patch)
// print(dTrain!.pointee)

// print(major)
// defer {
//     free(dTrain)
// }

// print(a)

// var dmx = UnsafeMutablePointer<DMatrixHandle?>(&dTrain)
// print(dTrain)
// dump(dTrain)
// var pNRow: UInt64 = 0
// var pNCol: UInt64 = 0
// let b = XGDMatrixNumRow(dTrain, &pNRow)
// let c = XGDMatrixNumCol(dTrain, &pNCol)

// var nRow: String
// = UnsafeMutablePointer<UInt64>(&pNRow)
// dump(pNRow, to: &nRow)
// print(b)
// print(pNRow)
// print(pNCol)

// var booster: BoosterHandle?
// let d = XGBoosterCreate(&dTrain, 1, &booster)
// print(d)

var dms = [dTrain]
var booster = BoosterCreate(dmHandles: &dms)
// var booster = BoosterCreate(dmHandle: &dTrain, lenDm: 1)

var e: Int32 = 0
// e = XGBoosterSetParam(booster, "versbosity", "1")
BoosterSetParam(handle: booster!, key: "verbosity", value: "3")
// e = XGBoosterSetParam(booster, "eval_metric", "logloss")
e = XGBoosterSetParam(booster, "max_depth", "9")
e = XGBoosterSetParam(booster, "eta", "1")
e = XGBoosterSetParam(booster, "objective", "binary:logistic")
print(e)
// print(dTrain!.pointee)

var grad: Float? = 0
var hess: Float? = 0
var len: UInt64 = 1
// let f = XGBoosterBoostOneIter(&booster, dTrain, &grad!, &hess!, len)
// guard XGBoosterUpdateOneIter(booster, 0, dTrain) >= 0 else {
//     print("boost update wrong")
//     exit(-1)
// }
BoosterUpdateOneIter(handle: booster!, nIter: 4, dmHandle: dTrain!)

let nTrees: Int = 10
// var evalResult: String?
// var evalResult: UnsafePointer<Int8>?
// // var evalNames: [UnsafePointer<Int8>?] = [UnsafePointer<Int8>("train"), UnsafePointer<Int8>("test")]
// var evalNames: UnsafeMutablePointer<UnsafePointer<Int8>?> = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 2)
// // defer {
// //     // free(&evalNames)
// //     evalNames.deallocate(capacity: 2)
// // }
// evalNames[0] = UnsafePointer<Int8>("train")
// evalNames[1] = UnsafePointer<Int8>("test")

let evalNames = ["train", "test"]
// evalNames.append("train")
// var evalNames = Array(String)
// var pEvalNames: UnsafeMutablePointer<UnsafePointer<UInt8>?> = UnsafeMutablePointer<UnsafePointer<UInt8>?>.allocate(capacity: evalNames.count)
// pEvalNames.initialize(from: evalNames.map { $0.utf8 }, count:
// evalNames.count)
// var pEvalNames = evalNames.map { $0.data(using: String.Encoding.utf8) }

var evalDm = [dTrain, dTest]

for i in 0 ..< nTrees {
    // XGBoosterUpdateOneIter(booster, i, dTrain)
    BoosterUpdateOneIter(handle: booster!, nIter: i, dmHandle: dTrain!)
    // XGBoosterEvalOneIter(booster, Int32(i), &evalDm, evalNames, 2, &evalResult)
    // print(String(cString: evalResult!))
    let evalResult = BoosterEvalOneIter(handle: booster!, nIter: i, dmHandle: &evalDm, evalNames: evalNames)
    print("evalResult: \(evalResult)")
    // let len = evalResult!.withMemoryRebound(to: Int8.self, capacity: 1) {
    //     // strlen($0)
    //     // String($0)
    //     $0
    // }
    // print("len \(len)")
    // print(evalResult!.pointee)
    // var result: UnsafePointer<Float>?
    // var outLen: UInt64 = 0
    // XGBoosterPredict(booster, dTest, 0, 0, 0, &outLen, &result)
    // print(result!.pointee)
}

var out_result: UnsafePointer<Float>?
// var outLen: UInt64 = 0
// XGBoosterPredict(booster, dTest, 0, 0, 0, &outLen, &result)
// print("outlen: \(outLen)")
// for _ in 0 ..< 20 {
//     // print(result!.pointee)
// }

let result = BoosterPredict(handle: booster!, dmHandle: dTest!, optionMask: 0,
                            nTreeLimit: 0, training: false)
print("result: len: \(result!.count), top5: \(result![0 ..< 5])")

var out_len: UInt64 = UInt64(0)
// var out_len: UInt64 = UInt64(result!.count)

// XGDMatrixGetFloatInfo(dTest, "label", &out_len, &out_result)
// print("y_test: ")
// for i in 0 ..< 5 {
//     print("%1.4f : \(out_result![Int(i)])")
// }

// print(out_len)
let trueLabels = DMatrixGetFloatInfo(handle: dTest!, label: "label")
print("trueLabels cnt: \(trueLabels!.count)")

// let err = XGBGetLastError()
// print(err!.pointee)

BoosterFree(booster!)
DMatrixFree(dTrain!)
DMatrixFree(dTest!)

print("Hello, world!")

// var internalImage = gdImageCreateTrueColor(Int32(10), Int32(20))
// print(internalImage!.pointee.sx)

let train = DMatrix(filename: "data/agaricus.txt.train", silent: false)
let test = DMatrix(filename: "data/agaricus.txt.test", silent: false)
print(train.shape)

let param = [
    "objective": "binary:logistic",
    "max_depth": "90",
    "verbosity": "3",
]
let bst = xgboost(data: train, numRound: 10, param: param, evalMetric: ["auc"])
print(bst)
print(bst.predict(data: test)[0 ..< 5])

print(BoosterSaveJsonConfig(handle: bst.handle!)!)
