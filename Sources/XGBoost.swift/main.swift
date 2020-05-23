import Cxgb

print("Hello, world!")

/// use macro to safely call fn
// func safeCall(fn)

// var dTrain: DMatrixHandle?
// var dTest: DMatrixHandle?

// var a = XGDMatrixCreateFromFile("data/agaricus.txt.train", 1, &dTrain)
// XGDMatrixCreateFromFile("data/agaricus.txt.test", 0, &dTest)

var dTrain = DMatrixFromFile(name: "data/agaricus.txt.train")
var dTest = DMatrixFromFile(name: "data/agaricus.txt.test")
// print(trainMat)

let nRow = DMatrixNumRow(&dTrain!)
print(nRow!)
let nCol = DMatrixNumCol(&dTrain!)
print(nCol!)

// var major: UnsafeMutablePointer<Int32>?
// var minor: UnsafeMutablePointer<Int32>?
// var patch: UnsafeMutablePointer<Int32>?
// // var major: Int32?
// // var minor: Int32?
// // var patch: Int32?

// XGBoostVersion(major, minor, patch)
// print(dTrain!.pointee)

// print(major)
defer {
    free(dTrain)
}

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

var booster = BoosterCreate(dmHandle: &dTrain, lenDm: 1)

var e: Int32 = 0
// e = XGBoosterSetParam(booster, "versbosity", "1")
BoosterSetParam(handle: &booster!, key: "verbosity", value: "1")
// e = XGBoosterSetParam(booster, "eval_metric", "logloss")
e = XGBoosterSetParam(booster, "max_depth", "2")
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
BoosterUpdateOneIter(handle: &booster!, nIter: 4, dmHandle: dTrain!)

let nTrees: Int = 10
// var evalResult: String?
// var evalResult: UnsafePointer<String>?
// var evalNames = ["train", "test"]
// var pEvalNames: UnsafeMutablePointer<UInt8>? = UnsafeMutablePointer<UInt8>.allocate(capacity: evalNames.count)
// pEvalNames.initialize(from: &evalNames, count: evalNames.count)

for i in 0 ..< nTrees {
    // XGBoosterUpdateOneIter(booster, i, dTrain)
    BoosterUpdateOneIter(handle: &booster!, nIter: i, dmHandle: dTrain!)
    // XGBoosterEvalOneIter(booster, i, &dTrain, &pEvalNames, 1, &evalResult)
    var result: UnsafePointer<Float>?
    var outLen: UInt64 = 0
    XGBoosterPredict(booster, dTest, 0, 0, 0, &outLen, &result)
    print(result!.pointee)
}

var result: UnsafePointer<Float>?
var outLen: UInt64 = 0
XGBoosterPredict(booster, dTest, 0, 0, 0, &outLen, &result)
for _ in 0 ..< 10 {
    print(result!.pointee)
}

let err = XGBGetLastError()
print(err!.pointee)

print("Hello, world!")

// var internalImage = gdImageCreateTrueColor(Int32(10), Int32(20))
// print(internalImage!.pointee.sx)
