import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(XGBoost_swiftTests.allTests),
    ]
}
#endif
