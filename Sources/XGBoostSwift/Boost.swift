import Cxgb
import Foundation

public typealias Param = [String: String]

public class XGBooster {
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
            debugLog("deinit XGBooster")
            BoosterFree(handle!)
        }
    }

    public func save(fname: String) throws {
        guard handle != nil else {
            errLog("booster not initialized!")
            return
        }
        try BoosterSaveModel(handle: handle!, fname: fname)
    }

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

    public func predict(data: DMatrix, outputMargin: Bool = false,
                        nTreeLimit: UInt = 0) -> [Float] {
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

    public func saveConfig(fname: String) {
        guard handle != nil else {
            errLog("booster not initialized!")
            return
        }

        let conf = BoosterSaveJsonConfig(handle: self.handle!)
        guard conf != nil else {
            errLog("Get json config failed!")
            return
        }
        let data: Data? = conf!.data(using: .utf8)
        let ok = FileManager().createFile(atPath: fname, contents: data)
        if !ok { errLog("save json config failed!") }
    }
}

// TODO: better way other than exit() when error
public func xgboost(data: DMatrix, numRound: Int = 10, param: Param = [:],
                    evalMetric: [String] = [], modelFile: String? = nil) -> XGBooster {
    if data.dmHandle == nil { exit(1) }

    var dms = [data.dmHandle]
    let booster = BoosterCreate(dmHandles: &dms)

    if booster == nil { exit(1) }

    if modelFile != nil {
        do {
            try BoosterLoadModel(handle: booster!, fname: modelFile!)
            debugLog("modle loeaded from file")
        } catch XGBoostError.modelLoadError {
            errLog("Error when loading model from file!")
            exit(1)
        } catch {
            errLog("Unknown error when loading model from file!")
            exit(1)
        }
    }

    let bst = XGBooster(handle: booster!)

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
    var booster: XGBooster
    let train: DMatrix
    let test: DMatrix

    init(train: DMatrix, test: DMatrix) {
        self.train = train
        self.test = test

        var dms = [self.train.dmHandle, self.test.dmHandle]
        // let handle = BoosterCreate(dmHandles: &dms)!
        self.booster = XGBooster(dms: &dms)
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
    var idxSet = [Int32](0 ..< Int32(data.nRow))

    if shuffle {
        idxSet = [Int32](idxSet.shuffled())
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

extension Array where Element: FloatingPoint {
    func sum() -> Element {
        return self.reduce(0, +)
    }

    func mean() -> Element {
        return self.sum() / Element(self.count)
    }

    func std() -> Element {
        let mean = self.mean()
        let v = self.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        let vari = v / (Element(self.count) - 1)
        return vari.squareRoot()
    }
}

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
        // let mean = v.reduce(0,+) / Float(v.count)
        let k = String(idxKey.split(separator: "\t")[1])
        results.append((k, v.mean(), v.std()))
    }
    return results
}

public typealias CVResult = [String: [Float]]

public func xgboostCV(data: DMatrix, nFold: Int = 5, numRound: Int = 10,
                      param: Param = [:],
                      evalMetric: [String] = [],
                      modelFile: String? = nil) -> CVResult {
    // handle metrics
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
