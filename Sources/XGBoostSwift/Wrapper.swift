import Cxgb

// TODO: check xgb lib loaded

extension String {
    func makeCString() -> UnsafePointer<Int8>? {
        // + 1 for the null-termination byte
        let cnt = self.utf8.count + 1
        let cstr = UnsafeMutablePointer<Int8>.allocate(capacity: cnt)
        self.withCString { baseAddr in
            cstr.initialize(from: baseAddr, count: cnt)
        }
        return UnsafePointer(cstr)
    }
}

enum XGBoostError: Error {
    case callError(errMsg: String)
    case unknownError(errMsg: String)
    case modelSaveError(errMsg: String)
    case modelLoadError(errMsg: String)
}

func lastError() -> String {
    return String(cString: XGBGetLastError())
}

func LogErrMsg(_ msg: String) {
    let errMsg = lastError()
    print("\(msg): \(errMsg)")
}

func check(call: () -> Int32, extraMsg: String = "") throws {
    if call() < 0 {
        let errMsg = "\(extraMsg),    XGBoost error: \(lastError())"
        throw XGBoostError.callError(errMsg: errMsg)
    }
}

public func xgboostVersion() -> (major: Int, minor: Int, patch: Int) {
    var major: Int32 = 0, minor: Int32 = 0, patch: Int32 = 0
    XGBoostVersion(&major, &minor, &patch)
    return (Int(major), Int(minor), Int(patch))
}

func DMatrixFromFile(name fname: String, silent: Bool = true) throws -> DMatrixHandle? {
    var silence: Int32 = 0
    if silent {
        silence = 1
    }

    var handle: DMatrixHandle?
    try check(call: { XGDMatrixCreateFromFile(fname, silence, &handle) },
              extraMsg: "Error when call XGDMatrixFromFile")

    return handle
}

func DMatrixFromMatrix(values: inout [Float], nRow: UInt64, nCol: UInt64,
                       missingVal: Float, nThread: Int32) throws -> DMatrixHandle? {
    var handle: DMatrixHandle?
    try check(call: { XGDMatrixCreateFromMat_omp(&values, nRow, nCol, missingVal,
                                                 &handle, nThread) },
              extraMsg: "Error when call XGDMatrixFromMat_omp")

    return handle
}

func DMatrixSaveBinary(handle: DMatrixHandle, fname: String,
                       silent: Bool = true) throws {
    var silence: Int32 = 0
    if silent {
        silence = 1
    }

    try check(call: { XGDMatrixSaveBinary(handle, fname, silence) },
              extraMsg: "Error when call XGDMatrixFromMat_omp")
}

func DMatrixFree(_ handle: DMatrixHandle) {
    guard XGDMatrixFree(handle) >= 0 else {
        let errMsg = lastError()
        print("free dmatrix failed, err msg: \(errMsg)")
        return
    }
}

func DMatrixNumRow(_ handle: DMatrixHandle) -> UInt64? {
    var nRow: UInt64 = 0
    guard XGDMatrixNumRow(handle, &nRow) >= 0 else {
        let errMsg = lastError()
        print("Get number of rows failed, err msg: \(errMsg)")
        return nil
    }
    return nRow
}

func DMatrixNumCol(_ handle: DMatrixHandle) -> UInt64? {
    var nCol: UInt64 = 0
    guard XGDMatrixNumCol(handle, &nCol) >= 0 else {
        let errMsg = lastError()
        print("Get number of cols failed, err msg: \(errMsg)")
        return nil
    }
    return nCol
}

func DMatrixGetFloatInfo(handle: DMatrixHandle, field: String) -> [Float] {
    var result: UnsafePointer<Float>?
    var len: UInt64 = 0

    guard XGDMatrixGetFloatInfo(handle, field, &len, &result) >= 0 else {
        let errMsg = lastError()
        print("Get dmatrix float info failed, err msg: \(errMsg)")
        return [Float]()
    }

    let buf = UnsafeBufferPointer(start: result, count: Int(len))
    return [Float](buf)
}

func DMatrixSetFloatInfo(handle: DMatrixHandle, field: String, data: [Float]) {
    guard XGDMatrixSetFloatInfo(handle, field, data, UInt64(data.count)) >= 0 else {
        let errMsg = lastError()
        print("Set dmatrix float info failed, err msg: \(errMsg)")
        return
    }
}

func DMatrixGetUIntInfo(handle: DMatrixHandle, field: String) -> [UInt] {
    var result: UnsafePointer<UInt32>?
    var len: UInt64 = 0

    guard XGDMatrixGetUIntInfo(handle, field, &len, &result) >= 0 else {
        let errMsg = lastError()
        print("Get dmatrix uint info failed, err msg: \(errMsg)")
        return [UInt]()
    }

    let buf = UnsafeBufferPointer(start: result, count: Int(len))
    return [UInt32](buf).map { UInt($0) }
}

func DMatrixSetUIntInfo(handle: DMatrixHandle, field: String, data: [UInt]) {
    guard XGDMatrixSetUIntInfo(handle, field, data.map { UInt32($0) },
                               UInt64(data.count)) >= 0 else {
        let errMsg = lastError()
        print("Set dmatrix uint info failed, err msg: \(errMsg)")
        return
    }
}

func DMatrixSliceDMatrix(_ handle: DMatrixHandle, idxSet: [Int]) -> DMatrixHandle? {
    let len: UInt64 = UInt64(idxSet.count)
    var idxs: [Int32] = idxSet.map { Int32($0) }
    var newHandle: DMatrixHandle?

    guard XGDMatrixSliceDMatrix(handle, &idxs, len, &newHandle) >= 0 else {
        LogErrMsg("Error when slice dmatrix")
        return nil
    }
    return newHandle
}

func DMatrixSliceDMatrixEx(_ handle: DMatrixHandle, idxSet: [Int], allowGroups: Bool) -> DMatrixHandle? {
    let len: UInt64 = UInt64(idxSet.count)
    var idxs: [Int32] = idxSet.map { Int32($0) }
    var newHandle: DMatrixHandle?

    var allowGrp: Int32 = 0
    if allowGroups {
        allowGrp = 1
    }

    guard XGDMatrixSliceDMatrixEx(handle, &idxs, len, &newHandle, allowGrp) >= 0 else {
        LogErrMsg("Error when slice dmatrix ex")
        return nil
    }
    return newHandle
}

func BoosterCreate(dmHandles: inout [DMatrixHandle?]) -> BoosterHandle? {
    let lenDm: UInt64 = UInt64(dmHandles.count)
    var handle: BoosterHandle?

    guard XGBoosterCreate(&dmHandles, lenDm, &handle) >= 0 else {
        let errMsg = lastError()
        print("create booster failed, err msg: \(errMsg)")
        return nil
    }
    return handle
}

// func BoosterCreate(dmHandle: inout DMatrixHandle?, lenDm: UInt64) -> BoosterHandle? {
//     var handle: BoosterHandle?
//     guard XGBoosterCreate(&dmHandle, lenDm, &handle) >= 0 else {
//         let errMsg = lastError()
//         print("create booster failed, err msg: \(errMsg)")
//         return nil
//     }
//     return handle
// }

func BoosterFree(_ handle: BoosterHandle) {
    guard XGBoosterFree(handle) >= 0 else {
        let errMsg = lastError()
        print("free booster failed, err msg: \(errMsg)")
        return
    }
}

func BoosterSetParam(handle: BoosterHandle, key: String, value: String) {
    guard XGBoosterSetParam(handle, key, value) >= 0 else {
        let errMsg = lastError()
        print("create booster failed, err msg: \(errMsg)")
        return
    }
}

func BoosterSetAttr(handle: BoosterHandle, key: String, value: String?) {
    guard XGBoosterSetAttr(handle, key, value) >= 0 else {
        let errMsg = lastError()
        print("set booster attr, err msg: \(errMsg)")
        return
    }
}

func BoosterGetAttr(handle: BoosterHandle, key: String) -> String? {
    var out: UnsafePointer<Int8>?
    var success: Int32 = -1

    guard XGBoosterGetAttr(handle, key, &out, &success) >= 0 else {
        return nil
    }

    guard success >= 0 else { return nil }

    return String(cString: out!)
}

func BoosterGetAttrNames(handle: BoosterHandle) -> [String]? {
    var len: UInt64 = 0
    // var out: [String?]
    // var out = [UnsafeMutablePointer<UnsafePointer<Int8>?>?]()
    var out = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

    guard XGBoosterGetAttrNames(handle, &len, out) >= 0 else {
        return nil
    }

    guard len > 0 else { return nil }

    var attrs = [String]()
    // for attr in out {
    for i in 0 ..< len {
        let a = String(cString: out.pointee![Int(i)]!)
        attrs.append(a)
    }
    return attrs
}

func BoosterUpdateOneIter(handle: BoosterHandle, currentIter nIter: Int, dmHandle: DMatrixHandle) {
    let iter: Int32 = Int32(nIter)
    guard XGBoosterUpdateOneIter(handle, iter, dmHandle) >= 0 else {
        let errMsg = lastError()
        print("create booster failed, err msg: \(errMsg)")
        return
    }
}

func BoosterEvalOneIter(handle: BoosterHandle, currentIter nIter: Int,
                        dmHandle: inout [DMatrixHandle?],
                        evalNames: [String]) -> String {
    var names = evalNames.map { $0.makeCString() }
    var result: UnsafePointer<Int8>?

    guard XGBoosterEvalOneIter(handle, Int32(nIter), &dmHandle, &names,
                               UInt64(evalNames.count), &result) >= 0 else {
        let errMsg = lastError()
        return "booster eval one iter failed, err msg: \(errMsg)"
    }

    return String(cString: result!)
}

func BoosterPredict(handle: BoosterHandle, dmHandle: DMatrixHandle,
                    optionMask: Int, nTreeLimit: Int, training: Bool) -> [Float]? {
    let optioin: Int32 = Int32(optionMask)
    let treeLim: UInt32 = UInt32(nTreeLimit)
    var isTraining: Int32 = 0
    if training {
        isTraining = 1
    }

    var outLen: UInt64 = 0
    var result: UnsafePointer<Float>?
    // defer { result?.deallocate() }
    guard XGBoosterPredict(handle, dmHandle, optioin, treeLim, isTraining,
                           &outLen, &result) >= 0 else {
        let errMsg = lastError()
        print("create booster failed, err msg: \(errMsg)")
        return nil
    }
    // TODO: deal potential issue when outLen is bigger than Int
    let buf = UnsafeBufferPointer(start: result, count: Int(outLen))
    return [Float](buf)
}

func BoosterSaveModel(handle: BoosterHandle, fname: String) throws {
    guard XGBoosterSaveModel(handle, fname) >= 0 else {
        let errMsg = lastError()
        throw XGBoostError.modelSaveError(errMsg: errMsg)
    }
}

func BoosterLoadModel(handle: BoosterHandle, fname: String) throws {
    guard XGBoosterLoadModel(handle, fname) >= 0 else {
        let errMsg = lastError()
        throw XGBoostError.modelLoadError(errMsg: errMsg)
    }
}

func BoosterSaveJsonConfig(handle: BoosterHandle) -> String? {
    var len: UInt64 = 0
    var str: UnsafePointer<Int8>?

    guard XGBoosterSaveJsonConfig(handle, &len, &str) >= 0 else {
        let errMsg = lastError()
        print("save booster config as json string failed, err msg: \(errMsg)")
        return nil
    }
    let jsonStr = String(cString: str!)
    return jsonStr
}
