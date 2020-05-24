import Cxgb

class DMatrix {
    private var handle: DMatrixHandle?

    var dmHandle: DMatrixHandle? { handle }

    var nRow: UInt64 {
        if handle != nil {
            let n = DMatrixNumRow(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    var nCol: UInt64 {
        if handle != nil {
            let n = DMatrixNumCol(handle!)
            guard n != nil else { return 0 }
            return n!
        } else { return 0 }
    }

    var shape: [UInt64] { [nRow, nCol] }

    var labels: [Float]? { DMatrixGetFloatInfo(handle: handle!, label: "label") }

    init(filename: String, silent: Bool = true) {
        handle = DMatrixFromFile(name: filename, silent: silent)
    }

    deinit {
        if handle != nil {
            print("deinit DMatrix")
            DMatrixFree(handle!)
        }
    }
}
