import Cxgb
import Foundation

public typealias Param = [String: String]

public class XGBooster {
    internal var handle: BoosterHandle?

    internal init(handle: BoosterHandle) {
        self.handle = handle
    }

    // internal init(fname: String) throws {
    //     let handle = BoosterCreate(dmHandles: &dms)
    // }

    deinit {
        if handle != nil {
            print("deinit XGBooster")
            BoosterFree(handle!)
        }
    }

    public func save(fname: String) throws {
        guard handle != nil else {
            print("booster not initialized!")
            return
        }
        try BoosterSaveModel(handle: handle!, fname: fname)
    }

    public func predict(data: DMatrix, outputMargin: Bool = false, nTreeLimit: UInt = 0) -> [Float] {
        guard handle != nil else {
            print("booster not initialized!")
            return [Float]()
        }
        guard data.dmHandle != nil else {
            print("DMatrix not initialized!")
            return [Float]()
        }
        var option = 0
        if outputMargin { option = 1 }
        let result = BoosterPredict(handle: handle!, dmHandle: data.dmHandle!,
                                    optionMask: option,
                                    nTreeLimit: nTreeLimit, training: false)
        guard result != nil else {
            print("no result predicted")
            return [Float]()
        }
        return result!
    }

    public func saveConfig(fname: String) {
        guard handle != nil else {
            print("booster not initialized!")
            return
        }

        let conf = BoosterSaveJsonConfig(handle: self.handle!)
        guard conf != nil else {
            print("Get json config failed!")
            return
        }
        let data: Data? = conf!.data(using: .utf8)
        let ok = FileManager().createFile(atPath: fname, contents: data)
        if !ok { print("save json config failed!") }
    }
}

// TODO: better way other than exit() when error
public func XGBoost(data: DMatrix, numRound: Int = 10, param: Param = [:],
                    evalMetric: [String] = [], modelFile: String? = nil) -> XGBooster {
    if data.dmHandle == nil { exit(1) }

    var dms = [data.dmHandle]
    let booster = BoosterCreate(dmHandles: &dms)

    if booster == nil { exit(1) }

    if modelFile != nil {
        do {
            try BoosterLoadModel(handle: booster!, fname: modelFile!)
            print("modle loeaded from file")
        } catch XGBoostError.modelLoadError {
            print("Error when loading model from file!")
            exit(1)
        } catch {
            print("Unknown error when loading model from file!")
            exit(1)
        }
    }

    let bst = XGBooster(handle: booster!)

    for (k, v) in param {
        print(k, v)
        BoosterSetParam(handle: booster!, key: k, value: v)
    }
    for v in evalMetric {
        print(v)
        BoosterSetParam(handle: booster!, key: "eval_metric", value: v)
    }

    for i in 0 ..< numRound {
        print("Round: \(i)")
        BoosterUpdateOneIter(handle: booster!, nIter: numRound, dmHandle: data.dmHandle!)
    }
    return bst
}
