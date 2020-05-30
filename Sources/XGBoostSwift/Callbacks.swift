
// typealias Callback = ( CallbackEnv )

public protocol XGBCallback {
    /// To specify is the callback should be called before an iteration
    var beforeIteration: Bool { get }
    /// The function to be called
    func callback(env: CallbackEnv)
}

public struct CallbackEnv {
    var currentIter: Int
    var beginIter: Int
    var endIter: Int
}

public struct SimplePrintEvalution: XGBCallback {
    public let beforeIteration: Bool = false

    public var period: Int = 1
    // let showSTDV: Bool = false

    public init() {}
    public func callback(env: CallbackEnv) {
        if period == 0 { return }

        let i = env.currentIter
        if i % period == 0 || i + 1 == env.beginIter || i + 1 == env.endIter {
            print("Simple printer evaluation callback")
        }
    }
}
