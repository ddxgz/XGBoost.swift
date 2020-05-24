// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "XGBoost.swift",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products
        // in packages which this package depends on.
        /// Using XGBoost 1.1.0
        .systemLibrary(name: "Cxgb", pkgConfig: "xgboost", providers: [.brew(["xgboost"])]),
        .target(
            name: "XGBoost.swift",
            dependencies: ["Cxgb"]
        ),
        .testTarget(
            name: "XGBoost.swiftTests",
            dependencies: ["XGBoost.swift"]
        ),
    ]
)
