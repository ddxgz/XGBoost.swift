import Cxgb

/// DMatrix is a wrap of the internal data structure DMatrix that used by
/// XGBoost. You can construct from a file (libsvm (default) or csv) or [Float].
/// To load csv file, specify uri parameter 'path_to_csv?format=csv' or set the
/// `format` param to 'csv'.
public class DMatrix {
    // TODO: should guard handle to be non nil?
    private var handle: DMatrixHandle?

    var dmHandle: DMatrixHandle? { handle }

    /// If the DMatrix is initialized, by simply checking if the DMatrixHandle
    /// is non-nil.
    public var initialized: Bool { handle != nil }

    /// The number of rows in the DMatrix
    public var numRow: UInt64 {
        if handle != nil {
            let n = DMatrixNumRow(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    /// The number of cols in the DMatrix
    public var numCol: UInt64 {
        if handle != nil {
            let n = DMatrixNumCol(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    /// The shape of the DMatrix, [numRow, numCol]
    public var shape: [UInt64] { [numRow, numCol] }

    /// The labels of the DMatrix
    public var label: [Float] {
        get {
            DMatrixGetFloatInfo(handle: handle!, field: "label")
        }
        set { DMatrixSetFloatInfo(handle: handle!, field: "label", data: newValue)
        }
    }

    public var weight: [Float] {
        get {
            getFloatInfo(field: "weight")
        }
        set {
            setFloatInfo(field: "weight", data: newValue)
        }
    }

    /// Not used base_margin for conforming to Swift naming convention
    public var baseMargin: [Float] {
        get {
            getFloatInfo(field: "base_margin")
        }
        set {
            setFloatInfo(field: "base_margin", data: newValue)
        }
    }

    public var base_margin: [Float] {
        get { baseMargin }
        set { baseMargin = newValue }
    }

    public func setGroup(_ newValue: [UInt]) {
        setUIntInfo(field: "group", data: newValue)
    }

    private func _guardHandle() throws {
        guard handle != nil else {
            throw XGBoostError.unknownError(
                errMsg: "DMatrix handle is nil, XGBoost error: \(lastError())")
        }
    }

    /// Construct DMatrix from file
    public init(fromFile fname: String,
                format: String = "libsvm",
                label: [Float]? = nil,
                weight: [Float]? = nil,
                baseMargin: [Float]? = nil,
                silent: Bool = true) throws {
        var name = fname
        if format.lowercased() == "csv", !fname.contains("format=csv") {
            name += "?format=csv"
        }
        try handle = DMatrixFromFile(name: name, silent: silent)
        self.setExtra(label: label, weight: weight, baseMargin: baseMargin)
    }

    /// Construct DMatrix from array of Float, by setting shape, missing values
    /// will be filled in automatically or by setting `missing` (Float.infinity
    /// as default).
    public init(fromArray array: [Float],
                shape: (row: Int, col: Int),
                label: [Float]? = nil,
                weight: [Float]? = nil,
                baseMargin: [Float]? = nil,
                missing NaValue: Float = -.infinity) throws {
        var values = array
        try handle = DMatrixFromMatrix(values: &values, nRow: UInt64(shape.row),
                                       nCol: UInt64(shape.col),
                                       missingVal: NaValue, nThread: 0)
        self.setExtra(label: label, weight: weight, baseMargin: baseMargin)
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
    public func saveBinary(toFile fname: String, silent: Bool = true) throws {
        try _guardHandle()
        try DMatrixSaveBinary(handle: handle!, fname: fname, silent: silent)
    }

    // TODO: accept differnt types of rows
    /// Slice the DMatrix by using an array of row indexes, return a DMatrix of
    /// the selected rows.
    public func slice(rows idxSet: [Int], allowGroups: Bool = false) -> DMatrix? {
        guard handle != nil else {
            errLog("dmatrix not initialized")
            return nil
        }
        let handle = DMatrixSliceDMatrixEx(self.handle!, idxSet: idxSet,
                                           allowGroups: allowGroups)
        return DMatrix(handle: handle)
    }

    /// Slice the DMatrix by using a range of row indexes, return a DMatrix of
    /// the selected rows.
    public func slice(rows idxSet: Range<Int>, allowGroups: Bool = false) -> DMatrix? {
        return slice(rows: Array(idxSet), allowGroups: allowGroups)
    }

    func setExtra(label: [Float]? = nil, weight: [Float]? = nil, baseMargin: [Float]? = nil) {
        if label != nil {
            self.label = label!
        }
        if weight != nil {
            self.weight = weight!
        }
        if baseMargin != nil {
            self.baseMargin = baseMargin!
        }
    }

    func setFloatInfo(field: String, data: [Float]) {
        DMatrixSetFloatInfo(handle: handle!, field: field, data: data)
    }

    func getFloatInfo(field: String) -> [Float] {
        return DMatrixGetFloatInfo(handle: handle!, field: field)
    }

    func setUIntInfo(field: String, data: [UInt]) {
        DMatrixSetUIntInfo(handle: handle!, field: field, data: data)
    }

    func getUIntInfo(field: String) -> [UInt] {
        return DMatrixGetUIntInfo(handle: handle!, field: field)
    }
}
