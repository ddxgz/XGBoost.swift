
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
    func callback(env: CallbackEnv) throws
}

enum EarlyStopError: Error {
    case noEvalResultError(_ msg: String)
    case stopAt(bestIteration: Int)
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

    - Parameters:
        - period: how often to print the evaluation result
        - showSTD: whether to show the standard deviation
        - printPrefix: the prefix string for each printed message, default as
          "[Simple callback printer]"
 */
public struct SimplePrintEvalution: XGBCallback {
    public let beforeIteration: Bool = false

    public var period: Int = 1
    var showSTD: Bool = true

    public var printPrefix: String = "[Simple callback printer]"

    public init(period: Int = 1, showSTD: Bool = true, printPrefix: String? = nil) {
        self.period = period
        self.showSTD = showSTD
        if printPrefix != nil {
            self.printPrefix = printPrefix!
        }
    }

    public func callback(env: CallbackEnv) {
        if period == 0 { return }

        let i = env.currentIter
        if i % period == 0 || i + 1 == env.beginIter || i + 1 == env.endIter {
            var msg = "\(printPrefix)\tcurrentIter:\(i)\tbeginIter:" +
                "\(env.beginIter)\tendIter:\(env.endIter)"
            if env.evalResult != nil {
                msg += "\tevalResult:"
                // for res in env.evalResult! {
                //     // msg += "  \(res)"
                //     msg += " (\(res.0): \(res.1)"
                //     if showSTD, res.2 != nil {
                //         msg += ", std: \(res.2!)"
                //     }
                //     msg += ") "
                // }
                msg += fmtMetric(result: env.evalResult!, showSTD: showSTD)
            }
            print(msg)
        }
    }
}

/// Callback for early stop
///  - Parameters:
///     - stoppingRounds: how many iterations allowed for no improving
///     - maximize: if to maximize the evaluation metric
public class EarlyStop: XGBCallback {
    public let beforeIteration: Bool = false

    public let stoppingRounds: Int

    let maximizeScore: Bool

    struct State {
        let maximizeScore: Bool
        var bestIteration: Int = 0
        var bestScore: Float
        var bestMsg: String
    }

    var state: State?

    public init(stoppingRounds: Int, maximize: Bool = false) {
        self.stoppingRounds = stoppingRounds
        self.maximizeScore = maximize
    }

    func initState(env: CallbackEnv) throws {
        var maximize = self.maximizeScore

        let bst = env.model

        if env.evalResult == nil {
            throw EarlyStopError.noEvalResultError("Need at least 1 set of evalSet")
        }

        if env.evalResult!.count > 1 {
            print("Only the 1st evaluation metric will be used for early stop!")
        }

        let metricLabel = env.evalResult!.last!.0
        let metric = metricLabel.split(separator: "-", maxSplits: 1).last!

        let maxMetrics = ["auc", "aucpr", "map", "ndcg"]
        for m in maxMetrics {
            if metric.split(separator: ":")[0] == m {
                maximize = true
                break
            }
        }

        let maxAtNMetrics = ["auc@", "aucpr@", "map@", "ndcg@"]
        for m in maxAtNMetrics {
            if metric.starts(with: m) {
                maximize = true
                break
            }
        }

        var bestScore = Float.infinity
        if maximize { bestScore = -.infinity }

        let msg = "[\(env.currentIter)]\t\(fmtMetric(result: env.evalResult!))"

        self.state = State(maximizeScore: maximize, bestScore: bestScore, bestMsg: msg)

        if bst != nil {
            if bst!.attr(key: "bestScore") != nil {
                self.state!.bestScore = Float(bst!.attr(key: "bestScore")!)!
                self.state!.bestIteration = Int(bst!.attr(key: "bestIteration")!)!
                self.state!.bestMsg = bst!.attr(key: "bestMsg")!
            } else {
                bst!.setAttr(key: "bestIteration", value: String(state!.bestIteration))
                bst!.setAttr(key: "bestScore", value: String(state!.bestScore))
            }
        } else {
            assert(env.cvPacks != nil)
        }
    }

    public func callback(env: CallbackEnv) throws {
        if state == nil {
            try initState(env: env)
        }

        let score = env.evalResult!.last!.1

        if (state!.maximizeScore && score > state!.bestScore) ||
            (!state!.maximizeScore && score < state!.bestScore) {
            let msg = "[\(env.currentIter)]\t\(fmtMetric(result: env.evalResult!))"
            print("msg in early stop callback: \(msg)")
            state!.bestMsg = msg
            state!.bestScore = score
            state!.bestIteration = env.currentIter

            if env.model != nil {
                env.model!.setAttr(key: "bestIteration", value: String(state!.bestIteration))
                env.model!.setAttr(key: "bestScore", value: String(state!.bestScore))
                env.model!.setAttr(key: "bestMsg", value: String(state!.bestMsg))
            }
        } else if (env.currentIter - state!.bestIteration) >= stoppingRounds {
            throw EarlyStopError.stopAt(bestIteration: state!.bestIteration)
        }
    }
}

func fmtMetric(result: EvalResult, showSTD: Bool = true) -> String {
    var msg = ""
    for res in result {
        // msg += "  \(res)"
        msg += "(\(res.0):\(res.1)"
        if showSTD, res.2 != nil {
            msg += ", std:\(res.2!)"
        }
        msg += ")\t"
    }
    return msg
}
