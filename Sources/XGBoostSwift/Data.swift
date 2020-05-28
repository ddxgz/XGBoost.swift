import Cxgb

/// DMatrix is a wrap of the internal data structure DMatrix that used by
/// XGBoost. You can construct from a file or [Float].
public class DMatrix {
    // should guard handle to be non nil?
    private var handle: DMatrixHandle?

    var dmHandle: DMatrixHandle? { handle }

    /// If the DMatrix is initialized, by simply checking if the DMatrixHandle
    /// is non-nil.
    public var initialized: Bool { handle != nil }

    /// The number of rows in the DMatrix
    public var nRow: UInt64 {
        if handle != nil {
            let n = DMatrixNumRow(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    /// The number of cols in the DMatrix
    public var nCol: UInt64 {
        if handle != nil {
            let n = DMatrixNumCol(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    /// The shape of the DMatrix, [nRow, nCol]
    public var shape: [UInt64] { [nRow, nCol] }

    /// The labels of the DMatrix
    public var labels: [Float]? { DMatrixGetFloatInfo(handle: handle!, label: "label") }

    private func _guardHandle() throws {
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "DMatrix handle is nil, XGBoost error: \(lastError())")
        }
    }

    /// Construct DMatrix from file
    public init(fname: String, silent: Bool = true) throws {
        // handle = DMatrixFromFile(name: fname, silent: silent)
        try handle = DMatrixFromFile(name: fname, silent: silent)
    }

    /// Construct DMatrix from array of Float
    public init(array: [Float], shape: (row: Int, col: Int),
                NaValue: Float = -.infinity) throws {
        var values = array
        try handle = DMatrixFromMatrix(values: &values, nRow: UInt64(shape.row),
                                       nCol: UInt64(shape.col),
                                       missingVal: NaValue, nThread: 0)
    }

    internal init(handle: DMatrixHandle?) {
        self.handle = handle
    }

    deinit {
        if handle != nil {
            debugLog("deinit DMatrix")
            DMatrixFree(handle!)
        }
    }

    /// Save the DMatrix to file
    public func save(fname: String, silent: Bool = true) throws {
        try _guardHandle()
        try DMatrixSaveBinary(handle: handle!, fname: fname, silent: silent)
    }

    // TODO: accept differnt types of rows
    /// Slice the DMatrix by using an array of row indexes, return a DMatrix of 
    /// the selected rows.
    public func slice(rows idxSet: [Int]) -> DMatrix? {
        guard handle != nil else {
            errLog("dmatrix not initialized")
            return nil
        }
        let handle = DMatrixSliceDMatrix(self.handle!, idxSet: idxSet)
        return DMatrix(handle: handle)
    }

    /// Slice the DMatrix by using a range of row indexes, return a DMatrix of 
    /// the selected rows.
    public func slice(rows idxSet: Range<Int>) -> DMatrix? {
        return slice(rows: Array(idxSet))
    }
}
