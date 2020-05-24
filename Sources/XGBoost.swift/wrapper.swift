import Cxgb

struct DMatrix {
    private var handle: DMatrixHandle?
}

struct XGBooster {
    private var handle: BoosterHandle?
}

func LastError() -> String {
    let err = XGBGetLastError()
    let errMsg = err!.pointee
    return String(errMsg)
}

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

func BoosterCreate(dmHandle: inout DMatrixHandle?, lenDm: UInt64) -> BoosterHandle? {
    var handle: BoosterHandle?
    guard XGBoosterCreate(&dmHandle, lenDm, &handle) >= 0 else {
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return nil
    }
    return handle
}

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

func BoosterUpdateOneIter(handle: BoosterHandle, nIter: Int, dmHandle: DMatrixHandle) {
    let iter: Int32 = Int32(nIter)
    guard XGBoosterUpdateOneIter(handle, iter, dmHandle) >= 0 else {
        let errMsg = LastError()
        print("create booster failed, err msg: \(errMsg)")
        return
    }
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
