import Cxgb
import Foundation

public typealias Param = [String: String]

/// A Booster of XGBoost, the model of XGBoost.
public class Booster {
    // TODO: guard handle non-nil
    internal var handle: BoosterHandle?

    /// If the Booster is initialized, by simply checking if the BoosterHandle
    /// is non-nil.
    public var initialized: Bool { handle != nil }

    internal init(handle: BoosterHandle) {
        self.handle = handle
    }

    internal init(dms: inout [DMatrixHandle?]) {
        self.handle = BoosterCreate(dmHandles: &dms)
        // let handle = BoosterCreate(dmHandles: &dms)
        // guard handle != nil else {
        //     throw XGBoostError.unknownError(
        //         errMsg: "Booster handle is nil when construction, XGBoost error: \(lastError())")
        // }
        // self.handle = handle
    }

    // internal init(fname: String) throws {
    //     let handle = BoosterCreate(dmHandles: &dms)
    // }

    public init(params: Param = Param(), cache: [DMatrix],
                modelFile: String? = nil) throws {
        var dms = cache.map { $0.dmHandle }
        let handle = BoosterCreate(dmHandles: &dms)
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "Booster handle is nil when construction, XGBoost error: \(lastError())")
        }
        self.handle = handle
        // try init(dms:dms)

        if modelFile != nil {
            try loadModel(fromFile: modelFile!)
        }

        if params.count > 0 {
            setParam(params)
        }
    }

    deinit {
        if handle != nil {
            debugLog("deinit Booster")
            BoosterFree(handle!)
        }
    }

    private func _guardHandle() throws {
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "Booster handle is nil, XGBoost error: \(lastError())")
        }
    }

    /// Use .json as filename suffix to save model to json.
    /// Refer to [XGBoost doc](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html)
    public func saveModel(toFile fname: String) throws {
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "Booster handle is nil, XGBoost error: \(lastError())")
        }
        try BoosterSaveModel(handle: handle!, fname: fname)
    }

    public func loadModel(fromFile fname: String) throws {
        // TODO: should create handle if nil?
        try BoosterLoadModel(handle: handle!, fname: fname)
    }

    /// Get attribute by key
    public func attr(key: String) -> String? {
        guard handle != nil else {
            errLog("booster not initialized!")
            return nil
        }
        return BoosterGetAttr(handle: handle!, key: key)
    }

    /// Get all the attributes
    public func attributes() -> [String: String] {
        var attributes = [String: String]()
        guard handle != nil else {
            errLog("booster not initialized!")
            return attributes
        }

        let attrNames = BoosterGetAttrNames(handle: handle!)
        guard attrNames != nil else {
            return attributes
        }

        for name in attrNames! {
            let att = attr(key: name)
            attributes[name] = att
        }
        return attributes
    }

    /// Set attribute, pass `value` as nil to delete an attribute.
    public func setAttr(key: String, value: String?) {
        guard handle != nil else {
            errLog("booster not initialized!")
            return
        }
        BoosterSetAttr(handle: handle!, key: key, value: value)
    }

    // TODO: accept eval_metric param input pass as a string of multiple values

    /**
     Set Parameters by key and value

      To set multiple `eval_metric`, use `setEvalMetrics`, or you can call
      setParam multiple times with different parameters.

      **Note**: XGBoost C API accepts wrong param key-value pairs when setting, but will
      throw error during training or evaluating. Check
      [document](https://xgboost.readthedocs.io/en/latest/parameter.html)
      to make sure they are right.

       **/
    public func setParam(key k: String, value v: String) {
        debugLog("Set param: \(k): \(v)")
        BoosterSetParam(handle: handle!, key: k, value: v)
    }

    /**
     Set Parameters by dictionary

      To set multiple `eval_metric`, use `setEvalMetrics`, or you can call
      setParam multiple times with different parameters.

      **Note**: XGBoost C API accepts wrong param key-value pairs when setting, but will
      throw error during training or evaluating. Check
      [document](https://xgboost.readthedocs.io/en/latest/parameter.html)
      to make sure they are right.

       **/
    public func setParam(_ params: [String: String]) {
        for (k, v) in params {
            debugLog("Set param: \(k): \(v)")
            BoosterSetParam(handle: handle!, key: k, value: v)
        }
    }

    /// Set eval_metric
    public func setEvalMetric(_ metrics: [String]) {
        for v in metrics {
            debugLog("Set param eval_metric: \(v)")
            setParam(key: "eval_metric", value: v)
        }
    }

    /// Update for 1 iteration
    public func update(data: DMatrix, currentIter: Int) {
        guard handle != nil else {
            errLog("booster not initialized!")
            return
        }
        guard data.dmHandle != nil else {
            errLog("data dmatrix not initialized!")
            return
        }
        BoosterUpdateOneIter(handle: self.handle!, currentIter: currentIter,
                             dmHandle: data.dmHandle!)
    }

    // TODO: func boost(dTrain: DMatrix, grad:[Float],hess:[Float]){}

    /**
     Evaluate a set of data
      - Parameters:
          - set: list of tuples (DMatrix, name of the eval data)
          - currentIter: current iteration
      - Returns: Evaluation result if successful, a string in a format like
        "[1]\ttrain-auc:0.938960\ttest-auc:0.948914",

     ```swift
     booster.eval(set: [(train, "train"),
                        (test, "test")], currentIter: 1)
     ```
      **/
    public func eval(set: [(DMatrix, String)], currentIter: Int) -> String? {
        guard handle != nil else {
            errLog("booster not initialized!")
            return nil
        }
        var dms = set.map { $0.0.dmHandle }
        let evalNames = set.map { $0.1 }
        return BoosterEvalOneIter(handle: handle!, currentIter: currentIter,
                                  dmHandle: &dms, evalNames: evalNames)
    }

    /**
     Evaluate data
      - Parameters:
          - data: DMatrix to be evaluate on
          - set: name of the eval data
          - currentIter: current iteration
      - Returns: Evaluation result if successful, a string in a format like
        "[1]\ttrain-auc:0.938960\ttest-auc:0.948914",

      **/
    public func eval(data: DMatrix, name: String, currentIter: Int = 0) -> String? {
        return eval(set: [(data, name)], currentIter: currentIter)
    }

    /**
     Predict labels on the data.
      - Parameters:
         - data: DMatrix - The data to predict on
         - outputMargin: bool - Whether to output the untransformed margin value
         - nTreeLimit: Int - Limit the number of trees, set to 0 to use all the
           trees (default value)
      - Returns: [Float]

      **/
    public func predict(data: DMatrix, outputMargin: Bool = false,
                        nTreeLimit: Int = 0) -> [Float] {
        guard handle != nil else {
            errLog("booster not initialized!")
            return [Float]()
        }
        guard data.dmHandle != nil else {
            errLog("DMatrix not initialized!")
            return [Float]()
        }
        var option = 0
        if outputMargin { option = 1 }
        let result = BoosterPredict(handle: handle!, dmHandle: data.dmHandle!,
                                    optionMask: option,
                                    nTreeLimit: nTreeLimit, training: false)
        guard result != nil else {
            errLog("no result predicted")
            return [Float]()
        }
        return result!
    }

    // TODO: provide option to force create dir when it doesn't exist
    public func saveConfig(toFile fname: String) throws {
        try _guardHandle()

        let conf = BoosterSaveJsonConfig(handle: self.handle!)
        guard conf != nil else {
            throw XGBoostError.unknownError(
                errMsg: "Get json config failed,  XGBoost error: \(lastError())")
        }
        let data: Data? = conf!.data(using: .utf8)
        let ok = FileManager().createFile(atPath: fname, contents: data)
        if !ok { errLog("save json config failed!") }
    }

    public func loadConfig(fromFile fname: String) throws {
        try _guardHandle()

        // let ok = FileManager().createFile(atPath: fname, contents: data)

        // let data: Data? = Data(contentsOf: URL(fname))
        var data = try String(contentsOfFile: fname)

        BoosterLoadJsonConfig(handle: self.handle!, json: &data)
    }
}

/**
 Train a booster with given parameters.
   - Parameters:
     - data: DMatrix
     - numRound: Int - Number of boosting iterations.
     - param: Dictionary - Booster parameters. If intend to use multiple
      `eval_metric`, they should be provided as the `evalMetric`.
     - evalMetric: [String] - to pass the `eval_metric` parameter to booster.
     - evalSet: list of tuples (DMatrix, name of the eval data). The
       validation sets will evaluated during training.
     - modelFile: String - If the modelFile param is provided, it will load the
       model from that file.

   - Returns: Booster

  **/
public func xgboost(params: Param = [:], data: DMatrix, numRound: Int = 10,
                    evalMetric: [String] = [],
                    evalSet: [(DMatrix, String)]? = nil, modelFile: String? = nil,
                    callbacks: [XGBCallback]? = nil) throws -> Booster {
    if data.dmHandle == nil {
        throw XGBoostError.unknownError(
            errMsg: "DMatrix handle is nil, XGBoost error: \(lastError())")
    }

    let evalset = evalSet ?? [(DMatrix, String)]()
    let evals = evalset.map { $0.0.dmHandle }

    var dms = [data.dmHandle] + evals
    let booster = BoosterCreate(dmHandles: &dms)

    if booster == nil {
        throw XGBoostError.unknownError(
            errMsg: "Booster handle is nil, XGBoost error: \(lastError())")
    }

    if modelFile != nil {
        do {
            try BoosterLoadModel(handle: booster!, fname: modelFile!)
            debugLog("modle loeaded from file")
        } catch XGBoostError.modelLoadError {
            errLog("Error when loading model from file!")
        } catch {
            errLog("Unknown error when loading model from file!")
        }
    }

    let bst = Booster(handle: booster!)

    bst.setParam(params)

    for v in evalMetric {
        debugLog("Set param eval_metric: \(v)")
        // BoosterSetParam(handle: booster!, key: "eval_metric", value: v)
        bst.setParam(key: "eval_metric", value: v)
    }

    for i in 0 ..< numRound {
        debugLog("Round: \(i)")

        if callbacks != nil {
            for callback in callbacks! {
                if callback.beforeIteration {
                    callback.callback(env: CallbackEnv(currentIter: i,
                                                       beginIter: 0,
                                                       endIter: numRound,
                                                       evalResult: nil))
                }
            }
        }

        // TODO: fix nIter as current iter
        // BoosterUpdateOneIter(handle: booster!, currentIter: i, dmHandle: data.dmHandle!)
        bst.update(data: data, currentIter: i)

        var evalResult = [(String, Float)]()
        if evalset.count > 0 {
            let evalMsg = bst.eval(set: evalset, currentIter: i)!
            let res = evalMsg.split(separator: "\t")[1...].map { $0.split(separator: ":") }
            evalResult = res.map { (String($0[0]), Float($0[1])!) }
        }

        if callbacks != nil {
            for callback in callbacks! {
                if !callback.beforeIteration {
                    callback.callback(env: CallbackEnv(currentIter: i,
                                                       beginIter: 0,
                                                       endIter: numRound,
                                                       evalResult: evalResult))
                }
            }
        }
    }
    return bst
}

internal struct CVPack {
    var booster: Booster
    let train: DMatrix
    let test: DMatrix

    init(train: DMatrix, test: DMatrix) {
        self.train = train
        self.test = test

        var dms = [self.train.dmHandle, self.test.dmHandle]
        // let handle = BoosterCreate(dmHandles: &dms)!
        self.booster = Booster(dms: &dms)
    }

    internal func update(_ round: Int) {
        // BoosterUpdateOneIter(handle: self.booster, nIter: round, dmHandle:
        // self.train)
        self.booster.update(data: train, currentIter: round)
    }

    internal func eval(_ round: Int) -> String? {
        // return self.booster.evalSet(dmHandle: [train, test],
        //                             evalNames: ["train", "test"], currentIter: round)
        return self.booster.eval(set: [(train, "train"), (test, "test")],
                                 currentIter: round)
    }
}

internal func makeNFold(data: DMatrix, nFold: Int = 5, param: Param = [:],
                        evalMetric: [String] = [], shuffle: Bool = true) -> [CVPack] {
    var cvpacks = [CVPack]()
    // var idxSet = [Int32](0 ..< Int32(data.nRow))
    var idxSet = [Int](0 ..< Int(data.nRow))

    if shuffle {
        // idxSet = [Int32](idxSet.shuffled())
        idxSet = [Int](idxSet.shuffled())
    }

    let foldSize = Int(data.nRow) / nFold
    for i in 0 ..< nFold {
        let testIdx = Array(idxSet[Int(i) * foldSize ..< (i + 1) * foldSize])
        let trainIdx = Array(Set(idxSet).subtracting(testIdx))
        let trainFold = data.slice(rows: trainIdx)
        let testFold = data.slice(rows: testIdx)
        cvpacks.append(CVPack(train: trainFold!, test: testFold!))
    }

    return cvpacks
}

typealias CvIterResult = [(String, Float, Float)]

func aggCV(_ results: [String?]) -> CvIterResult {
    var cvMap = [String: [Float]]()

    for (i, line) in results.enumerated() {
        if line == nil {
            errLog("Eval result of CV fold \(i) is empty")
            continue
        }
        let part = line!.split(separator: "\t")
        for (metricIdx, it) in part[1...].enumerated() {
            let kv = it.split(separator: ":")
            let k = kv[0]
            let v = kv[1]
            let idxKey = "\(metricIdx)\t\(k)"

            var values = cvMap[idxKey, default: [Float]()]
            values.append(Float(v)!)
            cvMap[idxKey] = values
        }
    }
    var results = CvIterResult()
    for (idxKey, v) in cvMap {
        let k = String(idxKey.split(separator: "\t")[1])
        results.append((k, v.mean(), v.std()))
    }
    return results
}

/// Each k, v pair is a measure's mean or std of each round
public typealias CVResult = [String: [Float]]

// TODO: support seed, early_stopping_rounds
/// Cross-validation with given parameters
public func xgboostCV(params: Param = [:], data: DMatrix,
                      numRound: Int = 10,
                      nFold: Int = 5,
                      evalMetric: [String] = []) -> CVResult {
    // TODO: handle metrics
    let cvFolds = makeNFold(data: data, nFold: nFold, param: params,
                            evalMetric: evalMetric, shuffle: true)

    var results = CVResult()
    for i in 0 ..< numRound {
        for fold in cvFolds {
            fold.update(i)
        }

        let res = aggCV(cvFolds.map { $0.eval(i) })
        for (k, mean, std) in res {
            var means = results[k + "-mean"] ?? [Float]()
            means.append(mean)
            results[k + "-mean"] = means

            var stds = results[k + "-std"] ?? [Float]()
            stds.append(std)
            results[k + "-std"] = stds
        }
    }

    return results
}
