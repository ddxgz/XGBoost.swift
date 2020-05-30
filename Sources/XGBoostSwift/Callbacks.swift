
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
    var evalResult: [(String, Float)]?
}

/**
 A simple callback printer

  **/
public struct SimplePrintEvalution: XGBCallback {
    public let beforeIteration: Bool = false

    public var period: Int = 1
    // let showSTDV: Bool = false

    public var printPrefix: String = "[Simple callback printer]"

    public init(period: Int = 1, printPrefix: String? = nil) {
        self.period = period
        if printPrefix != nil {
            self.printPrefix = printPrefix!
        }
    }

    public func callback(env: CallbackEnv) {
        if period == 0 { return }

        let i = env.currentIter
        if i % period == 0 || i + 1 == env.beginIter || i + 1 == env.endIter {
            var msg = "\(printPrefix)  currentIter: \(i)  beginIter: " +
                "\(env.beginIter)  endIter: \(env.endIter)"
            if env.evalResult != nil {
                msg += "  evalResult:"
                for res in env.evalResult! {
                    msg += "  \(res)"
                }
            }
            print(msg)
        }
    }
}
