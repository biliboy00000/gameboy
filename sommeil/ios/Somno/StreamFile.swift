import Foundation

/// Fichier binaire Float32 (little-endian) écrit au fil de la nuit, avec tampon.
final class StreamFile {
    let url: URL
    let columns: Int
    private let handle: FileHandle
    private var buffer = [Float]()
    private let lock = NSLock()
    private(set) var rows = 0

    init(url: URL, columns: Int) throws {
        self.url = url
        self.columns = columns
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        buffer.reserveCapacity(columns * 4096)
    }

    func append(_ row: [Float]) {
        precondition(row.count == columns)
        lock.lock()
        buffer.append(contentsOf: row)
        rows += 1
        let full = buffer.count >= columns * 3000
        lock.unlock()
        if full { flush() }
    }

    func appendMany(_ values: [Float]) {
        lock.lock()
        buffer.append(contentsOf: values)
        rows += values.count / columns
        let full = buffer.count >= columns * 3000
        lock.unlock()
        if full { flush() }
    }

    func flush() {
        lock.lock()
        let data = buffer.withUnsafeBufferPointer { Data(buffer: $0) }
        buffer.removeAll(keepingCapacity: true)
        lock.unlock()
        if !data.isEmpty { handle.write(data) }
    }

    func close() {
        flush()
        try? handle.close()
    }
}
