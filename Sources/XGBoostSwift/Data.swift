import Cxgb

public class DMatrix {
    // should guard handle to be non nil?
    private var handle: DMatrixHandle?

    var dmHandle: DMatrixHandle? { handle }

    public var initialized: Bool { handle != nil }

    public var nRow: UInt64 {
        if handle != nil {
            let n = DMatrixNumRow(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    public var nCol: UInt64 {
        if handle != nil {
            let n = DMatrixNumCol(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    public var shape: [UInt64] { [nRow, nCol] }

    public var labels: [Float]? { DMatrixGetFloatInfo(handle: handle!, label: "label") }

    public init(fname: String, silent: Bool = true) throws {
        // handle = DMatrixFromFile(name: fname, silent: silent)
        try handle = DMatrixFromFile(name: fname, silent: silent)
    }

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

    // TODO: accept differnt types of rows
    /// Use rows input is an array of row index to be selected
    public func slice(rows idxSet: [Int32]) -> DMatrix? {
        guard handle != nil else {
            errLog("dmatrix not initialized")
            return nil
        }
        let handle = DMatrixSliceDMatrix(self.handle!, idxSet: idxSet)
        return DMatrix(handle: handle)
    }
}
