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
    ("testCV", testCV),
    ("testBasic", testBasic),
  ]

  func testDMatrix() throws {
    let datafile = "data/agaricus.txt.train"
    let train = try DMatrix(fname: datafile)

    XCTAssertEqual(train.shape[0], 6513)
    XCTAssertEqual(train.shape[1], 126)

    // let csv = "data/train.csv?format=csv"
    let csv = "data/train.csv"
    let trainCSV = try DMatrix(fname: csv, format: "csv")
    XCTAssertEqual(trainCSV.shape[0], 892)
    XCTAssertEqual(trainCSV.shape[1], 12)

    let csv2 = "data/train.csv?format=csv"
    let trainCSV2 = try DMatrix(fname: csv, format: "csv")
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
    let base_marginSet = [Float](repeating: 1, count: Int(train.nRow))
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
    try trainSliced.save(fname: dmFile)
    let sliceLoaded = try DMatrix(fname: dmFile)
    XCTAssertEqual(trainSliced.shape[0], sliceLoaded.shape[0])
    XCTAssertEqual(trainSliced.shape[1], sliceLoaded.shape[1])

    let range = 0 ..< 100
    let mat = try DMatrix(array: range.map { _ in Float.random(in: 0 ..< 1) },
                          shape: (11, 10))
    XCTAssertTrue(mat.initialized)

    let matWithNa = try DMatrix(array: range.map { _ in Float.random(in: 0 ..< 1) },
                                shape: (21, 30))
    XCTAssertTrue(matWithNa.initialized)
  }

  func testBooster() throws {
    let train = try DMatrix(fname: "data/agaricus.txt.train")
    let test = try DMatrix(fname: "data/agaricus.txt.test")

    let param = [
      "objective": "binary:logistic",
      "max_depth": "2",
    ]
    let bst = try xgboost(data: train, numRound: 1, param: param, evalMetric: ["auc"])

    XCTAssertTrue(bst is Booster)

    let result = bst.predict(data: test)
    XCTAssertEqual(UInt64(result.count), test.nRow)

    let modelfile = "Tests/tmp/bst.model"
    try bst.save(fname: modelfile)
    let saved = FileManager().fileExists(atPath: modelfile)
    XCTAssertTrue(saved)

    let bstLoaded = try xgboost(data: train, numRound: 0, param: param,
                                evalMetric: ["auc"], modelFile: modelfile)
    let resultLoaded = bstLoaded.predict(data: test)
    XCTAssertTrue(resultLoaded.elementsEqual(result))

    let modelfileJson = "Tests/tmp/bst.json"
    try bst.save(fname: modelfileJson)
    let savedJson = FileManager().fileExists(atPath: modelfileJson)
    XCTAssertTrue(savedJson)

    let bstJsonLoaded = try xgboost(data: train, numRound: 0, param: param,
                                    evalMetric: ["auc"], modelFile: modelfileJson)
    let resultJsonLoaded = bstJsonLoaded.predict(data: test)
    XCTAssertTrue(resultJsonLoaded.elementsEqual(result))

    let configfile = "Tests/tmp/config.json"
    try bst.saveConfig(fname: configfile)
    let confSaved = FileManager().fileExists(atPath: configfile)
    XCTAssertTrue(confSaved)

    var lastEval: String = ""
    for i in 1 ... 5 {
      bst.update(data: train, currentIter: i)
      let evalResult = bst.evalSet(dmHandle: [train, test],
                                   evalNames: ["train", "test"], currentIter: i)

      let newEval = String(evalResult![evalResult!.index(evalResult!.startIndex, offsetBy: 4)...])
      XCTAssertNotEqual(lastEval, newEval)
      lastEval = newEval
    }
    let result2 = bst.predict(data: test)
    XCTAssertEqual(UInt64(result2.count), test.nRow)
    XCTAssertFalse(result2.elementsEqual(result))

    // for i in 0 ..< 5

    bst.setAttr(key: "key", value: "value")
    let attrs = bst.attributes()
    XCTAssertEqual(attrs.count, 1)
    bst.setAttr(key: "key", value: nil)
    let attrs2 = bst.attributes()
    XCTAssertEqual(attrs2.count, 0)
  }

  func testCV() throws {
    let train = try DMatrix(fname: "data/agaricus.txt.train")
    let param = [
      "objective": "binary:logistic",
      "max_depth": "9",
    ]
    // let cvFolds = XGBoostSwift.makeNFold(data: train, nFold: 5, evalMetric:
    // ["auc"], shuffle: true)
    let cvResults = xgboostCV(data: train, nFold: 5, numRound: 10, param: param)
    XCTAssertFalse(cvResults.isEmpty)
    XCTAssertEqual(cvResults.first!.value.count, 10)
  }

  func testBasic() throws {
    let ver = xgboostVersion()
    XCTAssertNotEqual(ver.major + ver.minor + ver.patch, 0)
  }
}
