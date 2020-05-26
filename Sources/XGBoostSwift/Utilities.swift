import Logging

var logger = Logger(label: "swift.XGBoost")

internal func errLog(_ msg: Logger.Message) {
    logger.error(msg)
}

internal func debugLog(_ msg: Logger.Message) {
    logger.debug(msg)
}
