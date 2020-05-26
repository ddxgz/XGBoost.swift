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
    ("testXGBooster", testXGBooster),
    ("testCV", testCV),
  ]

  func testDMatrix() throws {
    let datafile = "data/agaricus.txt.train"
    let train = DMatrix(fname: datafile)

    XCTAssertEqual(train.shape[0], 6513)
    XCTAssertEqual(train.shape[1], 126)

    let labels = train.labels
    XCTAssertNotNil(labels)
    XCTAssertEqual(train.shape[0], UInt64(labels!.count))

    let trainSliced = train.slice(rows: [0, 3])!
    XCTAssertEqual(trainSliced.shape[0], UInt64(2))
    XCTAssertEqual(trainSliced.shape[1], train.shape[1])
  }

  func testXGBooster() throws {
    let train = DMatrix(fname: "data/agaricus.txt.train")
    let test = DMatrix(fname: "data/agaricus.txt.test")

    let param = [
      "objective": "binary:logistic",
      "max_depth": "2",
    ]
    let bst = XGBoost(data: train, numRound: 1, param: param, evalMetric: ["auc"])

    XCTAssertTrue(bst is XGBooster)

    let result = bst.predict(data: test)
    XCTAssertEqual(UInt64(result.count), test.nRow)

    let modelfile = "Tests/tmp/bst.model"
    try bst.save(fname: modelfile)
    let saved = FileManager().fileExists(atPath: modelfile)
    XCTAssertTrue(saved)

    let bstLoaded = XGBoost(data: train, numRound: 0, param: param,
                            evalMetric: ["auc"], modelFile: modelfile)
    let resultLoaded = bstLoaded.predict(data: test)
    XCTAssertTrue(resultLoaded.elementsEqual(result))

    let configfile = "Tests/tmp/config.json"
    bst.saveConfig(fname: configfile)
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
  }

  func testCV() throws {
    let train = DMatrix(fname: "data/agaricus.txt.train")
    let param = [
      "objective": "binary:logistic",
      "max_depth": "9",
    ]
    // let cvFolds = XGBoostSwift.makeNFold(data: train, nFold: 5, evalMetric: ["auc"], shuffle: true)
  }
}
