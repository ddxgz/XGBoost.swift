import Cxgb
import Foundation

public typealias Param = [String: String]

/// A Booster of XGBoost, the model of XGBoost.
public class Booster {
    internal var handle: BoosterHandle?

    internal init(handle: BoosterHandle) {
        self.handle = handle
    }

    internal init(dms: inout [DMatrixHandle?]) {
        self.handle = BoosterCreate(dmHandles: &dms)
    }

    // internal init(fname: String) throws {
    //     let handle = BoosterCreate(dmHandles: &dms)
    // }

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
    public func save(fname: String) throws {
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "Booster handle is nil, XGBoost error: \(lastError())")
        }
        try BoosterSaveModel(handle: handle!, fname: fname)
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

    // func boost(dTrain: DMatrix, grad:[Float],hess:[Float]){}

    public func evalSet(dmHandle: [DMatrix],
                        evalNames: [String], currentIter: Int) -> String? {
        guard handle != nil else {
            errLog("booster not initialized!")
            return nil
        }
        var dms = dmHandle.map { $0.dmHandle }
        return BoosterEvalOneIter(handle: handle!, currentIter: currentIter,
                                  dmHandle: &dms, evalNames: evalNames)
    }

    /** Predict labels on the data.
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
    public func saveConfig(fname: String) throws {
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
}

/** Train a booster with given parameters.
  - Parameters:
    - data: DMatrix
    - numRound: Int - Number of boosting iterations.
    - param: Dictionary - Booster parameters. If intend to use multiple
     `eval_metric`, they should be provided as the `evalMetric`.
    - evalMetric: [String] - to pass the `eval_metric` parameter to booster.
    - modelFile: String - If the modelFile param is provided, it will load the
      model from that file.

  - Returns: Booster

 **/
public func xgboost(data: DMatrix, numRound: Int = 10, param: Param = [:],
                    evalMetric: [String] = [], modelFile: String? = nil) throws -> Booster {
    if data.dmHandle == nil {
        throw XGBoostError.unknownError(
            errMsg: "DMatrix handle is nil, XGBoost error: \(lastError())")
    }

    var dms = [data.dmHandle]
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

    for (k, v) in param {
        debugLog("Set param: \(k): \(v)")
        BoosterSetParam(handle: booster!, key: k, value: v)
    }
    for v in evalMetric {
        debugLog("Set param eval_metric: \(v)")
        BoosterSetParam(handle: booster!, key: "eval_metric", value: v)
    }

    for i in 0 ..< numRound {
        debugLog("Round: \(i)")
        // TODO: fix nIter as current iter
        BoosterUpdateOneIter(handle: booster!, currentIter: i, dmHandle: data.dmHandle!)
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
        return self.booster.evalSet(dmHandle: [train, test],
                                    evalNames: ["train", "test"], currentIter: round)
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

/// Cross-validation with given parameters
public func xgboostCV(data: DMatrix, nFold: Int = 5, numRound: Int = 10,
                      param: Param = [:],
                      evalMetric: [String] = []) -> CVResult {
    // TODO: handle metrics
    let cvFolds = makeNFold(data: data, nFold: nFold, param: param,
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
