import XCTest

import XGBoostSwift

final class XGBoostSwiftTests: XCTestCase {
    override func setUp() {
        try! FileManager().createDirectory(atPath: "Tests/tmp", withIntermediateDirectories: true)
    }

    override func tearDown() {
        try! FileManager().removeItem(atPath: "Tests/tmp")
    }

    static var allTests = [
        ("testDMatrix", testDMatrix),
        ("testBooster", testBooster),
        ("testBoosterSetParam", testBoosterSetParam),
        ("testBoosterPredict", testBoosterPredict),
        ("testCV", testCV),
        ("testBasic", testBasic),
        ("testCallback", testCallback),
        ("testFuncEval", testFuncEval),
        ("testFuncObj", testFuncObj),
        ("testImportance", testImportance),
    ]

    func testDMatrix() throws {
        let datafile = "data/agaricus.txt.train"
        let train = try DMatrix(fromFile: datafile)

        XCTAssertEqual(train.shape[0], 6513)
        XCTAssertEqual(train.shape[1], 126)

        // Load DMatrix from csv file
        // let csv = "data/train.csv?format=csv"
        let csv = "data/train.csv"
        let trainCSV = try DMatrix(fromFile: csv, format: "csv")
        XCTAssertEqual(trainCSV.shape[0], 892)
        XCTAssertEqual(trainCSV.shape[1], 12)

        let csv2 = "data/train.csv?format=csv"
        let trainCSV2 = try DMatrix(fromFile: csv2, format: "csv")
        XCTAssertEqual(trainCSV2.shape[0], 892)
        XCTAssertEqual(trainCSV2.shape[1], 12)

        let labels = train.label
        XCTAssertNotNil(labels)
        XCTAssertEqual(train.shape[0], UInt64(labels.count))

        let weights = train.weight
        XCTAssertEqual(weights.count, 0)
        let weightSet = [Float]([1, 3, 4])
        train.weight = weightSet
        let weightGet = train.weight
        XCTAssertTrue(weightSet.elementsEqual(weightGet))

        let base_margins = train.base_margin
        XCTAssertEqual(base_margins.count, 0)
        let base_marginSet = [Float](repeating: 1, count: Int(train.numRow))
        train.base_margin = base_marginSet
        let base_marginGet = train.base_margin
        XCTAssertTrue(base_marginSet.elementsEqual(base_marginGet))

        // let field1Info = [Float]([0, 1, 2, 3])
        // train.setFloatInfo(field: "weight", data: field1Info)
        // let field1 = train.getFloatInfo(field: "base_margin")
        // print(field1)
        // XCTAssertNotNil(field1)
        // XCTAssertTrue(field1Info.elementsEqual(field1!))
        // train.setFloatInfo(field: "field1", data: nil)
        // let field1Nil = train.getFloatInfo(field: "field1")
        // XCTAssertNotNil(field1Nil)

        let trainSliced = train.slice(rows: [0, 3])!
        XCTAssertEqual(trainSliced.shape[0], UInt64(2))
        XCTAssertEqual(trainSliced.shape[1], train.shape[1])

        let trainRanged = train.slice(rows: 0 ..< 10)!
        XCTAssertEqual(trainRanged.shape[0], UInt64(10))
        XCTAssertEqual(trainRanged.shape[1], train.shape[1])

        let trainSlicedGroup = train.slice(rows: [0, 3], allowGroups: true)!
        XCTAssertEqual(trainSlicedGroup.shape[0], UInt64(2))
        XCTAssertEqual(trainSlicedGroup.shape[1], train.shape[1])

        let dmFile = "Tests/tmp/dmFile.sliced"
        try trainSliced.saveBinary(toFile: dmFile)
        let sliceLoaded = try DMatrix(fromFile: dmFile)
        XCTAssertEqual(trainSliced.shape[0], sliceLoaded.shape[0])
        XCTAssertEqual(trainSliced.shape[1], sliceLoaded.shape[1])

        let range = 0 ..< 100
        let mat = try DMatrix(fromArray: range.map { _ in Float.random(in: 0 ..< 1) },
                              shape: (11, 10))
        XCTAssertTrue(mat.initialized)

        let matWithNa = try DMatrix(fromArray: range.map { _ in Float.random(in: 0 ..< 1) },
                                    shape: (21, 30))
        XCTAssertTrue(matWithNa.initialized)
    }

    func testBooster() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")
        let test = try DMatrix(fromFile: "data/agaricus.txt.test")

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
        ]
        let bst = try xgboost(params: param, data: train, numRound: 1)

        XCTAssertTrue(bst is Booster)

        let result = bst.predict(data: test)
        XCTAssertEqual(UInt64(result.count), test.numRow)

        let modelfile = "Tests/tmp/bst.model"
        try bst.saveModel(toFile: modelfile)
        let saved = FileManager().fileExists(atPath: modelfile)
        XCTAssertTrue(saved)

        let bstLoaded = try xgboost(params: param, data: train, numRound: 0,
                                    modelFile: modelfile)
        let resultLoaded = bstLoaded.predict(data: test)
        XCTAssertTrue(resultLoaded.elementsEqual(result))

        let modelfileJson = "Tests/tmp/bst.json"
        try bst.saveModel(toFile: modelfileJson)
        let savedJson = FileManager().fileExists(atPath: modelfileJson)
        XCTAssertTrue(savedJson)

        let bstJsonLoaded = try xgboost(params: param, data: train, numRound: 0,
                                        modelFile: modelfileJson)
        let resultJsonLoaded = bstJsonLoaded.predict(data: test)
        XCTAssertTrue(resultJsonLoaded.elementsEqual(result))

        let bstJsonLoaded2 = try Booster(params: param, cache: [train],
                                         modelFile: modelfileJson)
        let resultJsonLoaded2 = bstJsonLoaded2.predict(data: test)
        XCTAssertTrue(resultJsonLoaded2.elementsEqual(result))

        let configfile = "Tests/tmp/config.json"
        try bst.saveConfig(toFile: configfile)
        let confSaved = FileManager().fileExists(atPath: configfile)
        XCTAssertTrue(confSaved)

        var lastEval: String = ""
        for i in 1 ... 5 {
            bst.update(data: train, currentIter: i)
            // let evalResult = bPredictst.evalSet(dmHandle: [train, test],
            //                              evalNames: ["train", "test"], currentIter: i)
            let evalResult = bst.evalSet(evals: [(train, "train"), (test, "test")],
                                         currentIter: i)

            let newEval = String(evalResult![evalResult!.index(evalResult!.startIndex, offsetBy: 4)...])
            XCTAssertNotEqual(lastEval, newEval)
            lastEval = newEval
        }
        let result2 = bst.predict(data: test)
        XCTAssertEqual(UInt64(result2.count), test.numRow)
        XCTAssertFalse(result2.elementsEqual(result))

        // for i in 0 ..< 5

        bst.setAttr(key: "key", value: "value")
        let attrs = bst.attributes()
        XCTAssertEqual(attrs.count, 1)
        bst.setAttr(key: "key", value: nil)
        let attrs2 = bst.attributes()
        XCTAssertEqual(attrs2.count, 0)

        // Construct from DMatrix cache
        let bst2 = try Booster(params: param, cache: [train, test])
        XCTAssertTrue(bst2.initialized)
        let bst3 = try Booster(cache: [train, test])
        XCTAssertTrue(bst3.initialized)

        try bst.dumpModel(toFile: "Tests/tmp/testmodeldump.txt")
        try bst.dumpModel(toFile: "Tests/tmp/testmodeldump.json", dumpFormat: "json")
        try bst.dumpModel(toFile: "Tests/tmp/testmodeldump.dot", dumpFormat: "dot")
    }

    func testCV() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")
        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
        ]
        // let cvFolds = XGBoostSwift.makeNFold(data: train, nFold: 5, evalMetric:
        // ["auc"], shuffle: true)
        let cvResults = try xgboostCV(params: param, data: train, numRound: 10, nFold: 5)
        XCTAssertFalse(cvResults.isEmpty)
        XCTAssertEqual(cvResults.first!.value.count, 10)
    }

    func testBasic() throws {
        let ver = xgboostVersion()
        XCTAssertNotEqual(ver.major + ver.minor + ver.patch, 0)

        xgbRegisterLogCallback { print("xgb log callback: \(String(cString: $0!))") }

        let train = try DMatrix(fromFile: "data/agaricus.txt.train")

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
            ("verbosity", "3"),
        ]
        let bst = try xgboost(params: param, data: train, numRound: 5)
    }

    func testBoosterSetParam() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
            ("eval_metric", "auc"),
            ("eval_metric", "error"),
        ]
        let bst = try xgboost(params: param, data: train, numRound: 1)

        bst.setParam(name: "alpha", value: "0.1")
        bst.setEvalMetric(["logloss", "rmse"])

        // TODO: read json config file to check if it has the set params
        let configfile = "Tests/tmp/config.json"
        try bst.saveConfig(toFile: configfile)
        let confSaved = FileManager().fileExists(atPath: configfile)
        XCTAssertTrue(confSaved)

        try bst.loadConfig(fromFile: configfile)
    }

    func testCallback() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")
        let test = try DMatrix(fromFile: "data/agaricus.txt.test")

        var callbacks: [XGBCallback] = [SimplePrintEvalution(period: 1, showSTD: true)]
        let bst = try xgboost(data: train, numRound: 10,
                              evalSet: [(train, "train"), (test, "test")],
                              callbacks: callbacks)

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
            ("eval_metric", "auc"),
            // ("eval_metric", "error"),
        ]

        callbacks.append(EarlyStop(stoppingRounds: 1, maximize: true))
        let cvResults = try xgboostCV(params: param, data: train, numRound: 10,
                                      nFold: 5,
                                      callbacks: callbacks)
    }

    func testFuncEval() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")
        let test = try DMatrix(fromFile: "data/agaricus.txt.test")

        func dumEval(preds: [Float], dmatrix: DMatrix) -> (String, Float) {
            let labels = dmatrix.label
            let prob = preds.map { x -> Float in
                if x > 0 { return 1.0 } else { return 0.0 }
            }
            var cnt: Float = 0
            for (label, pred) in zip(labels, prob) {
                if label == pred {
                    cnt += 1
                }
            }
            return ("dumEval", cnt / Float(labels.count))
        }

        let callbacks = [SimplePrintEvalution(period: 1)]
        let bst = try xgboost(data: train, numRound: 10,
                              evalSet: [(train, "train"), (test, "test")],
                              fnEval: dumEval,
                              callbacks: callbacks)
    }

    func testFuncObj() throws {
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

        let callbacks = [SimplePrintEvalution(period: 1)]
        let bst = try xgboost(data: train, numRound: 10,
                              evalSet: [(train, "train"), (test, "test")],
                              fnObj: logLossObj,
                              fnEval: dumEval,
                              callbacks: callbacks)
    }

    func testBoosterPredict() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")
        let test = try DMatrix(fromFile: "data/agaricus.txt.test")

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
        ]
        let bst = try xgboost(params: param, data: train, numRound: 1)

        let result = bst.predict(data: test)
        XCTAssertTrue(result.reduce(true) { $0 && ($1 <= 1) })
        let result1 = bst.predict(data: test, outputMargin: true)
        XCTAssertFalse(result.elementsEqual(result1))
        let result3 = bst.predict(data: test, outputMargin: true, predLeaf: true)
        XCTAssertFalse(result3.elementsEqual(result1))
        let result4 = bst.predict(data: test, outputMargin: false, predLeaf: false,
                                  predContribs: true)
        let result5 = bst.predict(data: test, outputMargin: true, predLeaf: false,
                                  predContribs: true)
        XCTAssertTrue(result4.elementsEqual(result5))
    }

    func testImportance() throws {
        let train = try DMatrix(fromFile: "data/agaricus.txt.train")

        let param = [
            ("objective", "binary:logistic"),
            ("max_depth", "2"),
            ("verbosity", "3"),
        ]
        let bst = try xgboost(params: param, data: train, numRound: 5)
        let importance = try bst.getScore()
        XCTAssertTrue(importance.count != 0)
    }
}
