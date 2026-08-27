// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "rmlx-menu",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "rmlx-menu",
            targets: ["RmlxMenu"]
        )
    ],
    targets: [
        .executableTarget(
            name: "RmlxMenu"
        )
    ]
)
