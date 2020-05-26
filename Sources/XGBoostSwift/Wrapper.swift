import Cxgb

// TODO: check xgb lib loaded

enum XGBoostError: Error {
    case modelSaveError(errMsg: String)
    case modelLoadError(errMsg: String)
}

func LastError() -> String {
    let err = XGBGetLastError()
    let errMsg = String(cString: err!)
    return errMsg
}

func LogErrMsg(_ msg: String) {
    let errMsg = LastError()
    print("\(msg): \(errMsg)")
}

// func PrintIfError(_ err: Int) {
//     if err >=0 {
//         let errMsg = LastError()
//         print("create xgdmatrix from file failed, err msg: \(errMsg)")
//         return nil
//     }
// }

func DMatrixFromFile(name fname: String, silent: Bool = true) -> DMatrixHandle? {
    var silence: Int32 = 0
    if silent {
        silence = 1
    }

    // var dm = DMatrix()
    var handle: DMatrixHandle?
    guard XGDMatrixCreateFromFile(fname, silence, &handle) >= 0 else {
        let errMsg = LastError()
        print("create xgdmatrix from file failed, err msg: \(errMsg)")
        return nil
    }
    return handle
}

func DMatrixFree(_ handle: DMatrixHandle) {
    guard XGDMatrixFree(handle) >= 0 else {
        let errMsg = LastError()
        print("free dmatrix failed, err msg: \(errMsg)")
        return
    }
}

func DMatrixNumRow(_ handle: DMatrixHandle) -> UInt64? {
    var nRow: UInt64 = 0
    guard XGDMatrixNumRow(handle, &nRow) >= 0 else {
        let errMsg = LastError()
        print("Get number of rows failed, err msg: \(errMsg)")
        return nil
    }
    return nRow
}

func DMatrixNumCol(_ handle: DMatrixHandle) -> UInt64? {
    var nCol: UInt64 = 0
    guard XGDMatrixNumCol(handle, &nCol) >= 0 else {
        let errMsg = LastError()
        print("Get number of cols failed, err msg: \(errMsg)")
        return nil
    }
    return nCol
}

func DMatrixGetFloatInfo(handle: DMatrixHandle, label: String) -> [Float]? {
    var result: UnsafePointer<Float>?
    var len: UInt64 = 0
    guard XGDMatrixGetFloatInfo(handle, label, &len, &result) >= 0 else {
        let errMsg = LastError()
        print("Get dmatrix float info failed, err msg: \(errMsg)")
        return nil
    }

    let buf = UnsafeBufferPointer(start: result, count: Int(len))
    return [Float](buf)
}

func DMatrixSliceDMatrix(_ handle: DMatrixHandle, idxSet: [Int32]) -> DMatrixHandle? {
    let len: UInt64 = UInt64(idxSet.count)
    // var idxs: [UnsafeMutablePointer<Int32>] = idxSet.map {
    // UnsafeBufferPointer<Int32>($0) }
    var idxs: [Int32] = idxSet
    var newHandle: DMatrixHandle?
    guard XGDMatrixSliceDMatrix(handle, &idxs, len, &newHandle) >= 0 else {
        LogErrMsg("Error when slice dmatrix")
        return nil
    }
    return newHandle
}

func BoosterCreate(dmHandles: inout [DMatrixHandle?]) -> BoosterHandle? {
    let lenDm: UInt64 = UInt64(dmHandles.count)
    var handle: BoosterHandle?
    guard XGBoosterCreate(&dmHandles, lenDm, &handle) >= 0 else {
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return nil
    }
    return handle
}

// func BoosterCreate(dmHandle: inout DMatrixHandle?, lenDm: UInt64) -> BoosterHandle? {
//     var handle: BoosterHandle?
//     guard XGBoosterCreate(&dmHandle, lenDm, &handle) >= 0 else {
//         let errMsg = LastError()
//         print("create booster failed, err msg: \(errMsg)")
//         return nil
//     }
//     return handle
// }

func BoosterFree(_ handle: BoosterHandle) {
    guard XGBoosterFree(handle) >= 0 else {
        let errMsg = LastError()
        print("free booster failed, err msg: \(errMsg)")
        return
    }
}

func BoosterSetParam(handle: BoosterHandle, key: String, value: String) {
    guard XGBoosterSetParam(handle, key, value) >= 0 else {
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return
    }
}

func BoosterUpdateOneIter(handle: BoosterHandle, currentIter nIter: Int, dmHandle: DMatrixHandle) {
    let iter: Int32 = Int32(nIter)
    guard XGBoosterUpdateOneIter(handle, iter, dmHandle) >= 0 else {
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return
    }
}

extension String {
    func makeCString() -> UnsafePointer<Int8>? {
        // + 1 for the null-termination byte
        let cnt = self.utf8.count + 1
        let cstr = UnsafeMutablePointer<Int8>.allocate(capacity: cnt)
        self.withCString { baseAddr in
            cstr.initialize(from: baseAddr, count: cnt)
        }
        // return UnsafePointer<Int8>(cstr)
        return UnsafePointer(cstr)
    }
}

func BoosterEvalOneIter(handle: BoosterHandle, currentIter nIter: Int,
                        dmHandle: inout [DMatrixHandle?],
                        evalNames: [String]) -> String {
    // TODO: solve dangling pointer
    // var names: [UnsafePointer<Int8>?] = evalNames.map { UnsafePointer<Int8>($0) }
    // var names: [UnsafePointer<Int8>?] = evalNames.map { $0}
    // var names = evalNames.map { UnsafeBufferPointer<Int8>(start: &$0, count:
    // 1) }
    // var names = UnsafeMutablePointer<Int8>.allocate(capacity: evalNames.count)
    // names.initialize(from: &evalNames, count: evalNames.count)
    var names = evalNames.map { $0.makeCString() }

    // var dms:
    var result: UnsafePointer<Int8>?

    guard XGBoosterEvalOneIter(handle, Int32(nIter), &dmHandle, &names,
                               UInt64(evalNames.count), &result) >= 0 else {
        let errMsg = LastError()
        return "booster eval one iter failed, err msg: \(errMsg)"
    }

    return String(cString: result!)
}

func BoosterPredict(handle: BoosterHandle, dmHandle: DMatrixHandle,
                    optionMask: Int, nTreeLimit: UInt, training: Bool) -> [Float]? {
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
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return nil
    }
    // TODO: deal potential issue when outLen is bigger than Int
    let buf = UnsafeBufferPointer(start: result, count: Int(outLen))
    return [Float](buf)
}

/// throw error?
func BoosterSaveModel(handle: BoosterHandle, fname: String) throws {
    guard XGBoosterSaveModel(handle, fname) >= 0 else {
        let errMsg = LastError()
        throw XGBoostError.modelSaveError(errMsg: errMsg)
    }
}

func BoosterLoadModel(handle: BoosterHandle, fname: String) throws {
    guard XGBoosterLoadModel(handle, fname) >= 0 else {
        let errMsg = LastError()
        throw XGBoostError.modelLoadError(errMsg: errMsg)
    }
}

func BoosterSaveJsonConfig(handle: BoosterHandle) -> String? {
    var len: UInt64 = 0
    var str: UnsafePointer<Int8>?
    guard XGBoosterSaveJsonConfig(handle, &len, &str) >= 0 else {
        let errMsg = LastError()
        print("save booster config as json string failed, err msg: \(errMsg)")
        return nil
    }
    let jsonStr = String(cString: str!)
    return jsonStr
}
