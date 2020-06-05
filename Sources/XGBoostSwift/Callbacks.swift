
// typealias Callback = ( CallbackEnv )

/// protocol for Callback sent to training / cross validation function.
/// Custom callback need to implement the `callback()` method that has 1
/// parameter in type of `CallbackEnv`. A implementation of callback should also
/// have a property `beforeIteration` to specify whether the callback should be
/// executed before a iteration.
public protocol XGBCallback {
    /// To specify is the callback should be called before an iteration
    var beforeIteration: Bool { get }
    /// The function to be called
    func callback(env: CallbackEnv)
}

/**
 Environment used for callbacks, what can be accessed by method `callback()`
   - Parameters:
     - model: Booster? - the Booster used when calling callback in training,
         when calling `xgboost()`.
     - cvPacks: [CVPack]? - the CVPacks used when calling callback in cross
         validation, when calling `xgboostCV()`.
     - currentIter: Int
     - beginIter: Int
     - endIter: Int
     - evalResult: evaluation result during training, when calling `xgboost()`.
     - cvEvalResult: evaluation result during cross validation, when calling `xgboostCV()`.

 */
public struct CallbackEnv {
    var model: Booster? = nil
    var cvPacks: [CVPack]? = nil
    var currentIter: Int
    var beginIter: Int
    var endIter: Int
    var evalResult: EvalResult? = nil
}

/**
 A simple callback printer

 */
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
                msg += "  evalResult: "
                for res in env.evalResult! {
                    // msg += "  \(res)"
                    msg += " (\(res.0): \(res.1)"
                    if res.2 != nil {
                        msg += ", std: \(res.2!)"
                    }
                    msg += ") "
                }
            }
            print(msg)
        }
    }
}
