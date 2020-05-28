import Logging

var logger = Logger(label: "swift.XGBoost")

internal func errLog(_ msg: Logger.Message) {
    logger.error(msg)
}

internal func debugLog(_ msg: Logger.Message) {
    logger.debug(msg)
}

extension Array where Element: FloatingPoint {
    func sum() -> Element {
        return self.reduce(0, +)
    }

    func mean() -> Element {
        return self.sum() / Element(self.count)
    }

    func std() -> Element {
        let mean = self.mean()
        let v = self.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        let vari = v / (Element(self.count) - 1)
        return vari.squareRoot()
    }
}
