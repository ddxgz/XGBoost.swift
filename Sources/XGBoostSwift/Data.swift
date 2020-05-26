import Cxgb
import Logging

var logger = Logger(label: "swift.XGBoost")

public class DMatrix {
    private var handle: DMatrixHandle?

    var dmHandle: DMatrixHandle? { handle }

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

    public init(fname: String, silent: Bool = true) {
        handle = DMatrixFromFile(name: fname, silent: silent)
    }

    deinit {
        if handle != nil {
            logger.debug("deinit DMatrix")
            DMatrixFree(handle!)
        }
    }
}
