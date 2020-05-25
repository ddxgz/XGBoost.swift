import XCTest

import XGBoostSwift

final class XGBoostSwiftTests: XCTestCase {
  static var allTests = [
    ("testDMatrix", testDMatrix),
    ("testXGBooster", testXGBooster),
  ]

  func testDMatrix() throws {
    let datafile = "data/agaricus.txt.train"
    let train = DMatrix(filename: datafile, silent: false)

    XCTAssertEqual(train.shape[0], 6513)
    XCTAssertEqual(train.shape[1], 126)
  }

  func testXGBooster() throws {
    let train = DMatrix(filename: "data/agaricus.txt.train")
    let test = DMatrix(filename: "data/agaricus.txt.test")

    let param = [
      "objective": "binary:logistic",
      "max_depth": "90",
    ]
    let bst = XGBoost(data: train, numRound: 1, param: param, evalMetric: ["auc"])

    XCTAssertTrue(bst is XGBooster)

    let result = bst.predict(data: test)
    XCTAssertEqual(UInt64(result.count), test.nRow)
  }

  // func testBasic() throws {}
}
