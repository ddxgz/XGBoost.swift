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
  ]

  func testDMatrix() throws {
    let datafile = "data/agaricus.txt.train"
    let train = DMatrix(fname: datafile)

    XCTAssertEqual(train.shape[0], 6513)
    XCTAssertEqual(train.shape[1], 126)

    let labels = train.labels
    XCTAssertNotNil(labels)
    XCTAssertEqual(train.shape[0], UInt64(labels!.count))
  }

  func testXGBooster() throws {
    let train = DMatrix(fname: "data/agaricus.txt.train")
    let test = DMatrix(fname: "data/agaricus.txt.test")

    let param = [
      "objective": "binary:logistic",
      "max_depth": "90",
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
  }

  // func testBasic() throws {}
}
