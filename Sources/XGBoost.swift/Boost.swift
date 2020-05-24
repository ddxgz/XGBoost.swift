import Cxgb

typealias Param = [String: String]

class XGBooster {
    internal var handle: BoosterHandle?

    init(handle: BoosterHandle) {
        self.handle = handle
    }

    deinit {
        if handle != nil {
            print("deinit XGBooster")
            BoosterFree(handle!)
        }
    }

    func predict(data: DMatrix, outputMargin: Bool = false, nTreeLimit: UInt = 0) -> [Float] {
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

    // func save {}
    // func load {}
}

func XGBoost(data: DMatrix, param: Param = [:], numRound: Int,
             evalMetric: [String] = []) -> XGBooster {
    if data.dmHandle == nil { exit(1) }

    var dms = [data.dmHandle]
    let booster = BoosterCreate(dmHandles: &dms)

    if booster == nil { exit(1) }

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