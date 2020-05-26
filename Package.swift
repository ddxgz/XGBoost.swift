// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "XGBoostSwift",
    // platforms: [
    //     .macOS(.v10_15),
    // ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.2.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products
        // in packages which this package depends on.
        /// Using XGBoost 1.1.0
        .systemLibrary(name: "Cxgb", pkgConfig: "xgboost", providers: [.brew(["xgboost"])]),
        .target(
            name: "XGBoostSwift",
            dependencies: ["Cxgb", .product(name: "Logging", package: "swift-log")]
        ),
        // .target(
        //     name: "run",
        //     dependencies: ["XGBoostSwift"]
        // ),
        .testTarget(
            name: "XGBoostSwiftTests",
            dependencies: ["XGBoostSwift"]
        ),
    ]
)
