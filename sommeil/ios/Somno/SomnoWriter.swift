import Foundation

/// Écrit le format .somno : "SOMNO001" | UInt32 LE longueur | en-tête JSON | sections Float32.
/// Identique à l'export de l'app web (web/js/export.js) et lu par analysis/somno/io.py.
enum SomnoWriter {
    struct Section {
        let kind: String
        let file: StreamFile
        let rate: Double
        let columns: [String]
        var extra: [String: Any] = [:]
    }

    static func write(to url: URL, meta: [String: Any], sections: [Section]) throws {
        var descr = [[String: Any]]()
        var offset = 0
        for s in sections {
            let bytes = (try FileManager.default.attributesOfItem(atPath: s.file.url.path)[.size] as? Int) ?? 0
            let rowBytes = s.file.columns * 4
            let n = bytes / rowBytes
            var d: [String: Any] = ["kind": s.kind, "n": n, "cols": s.file.columns, "dtype": "float32",
                                    "offset": offset, "bytes": n * rowBytes, "rate": s.rate, "columns": s.columns]
            s.extra.forEach { d[$0.key] = $0.value }
            descr.append(d)
            offset += n * rowBytes
        }
        var m = meta
        m["sections"] = descr
        m["format"] = "somno/1"
        let head = try JSONSerialization.data(withJSONObject: m)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let out = try FileHandle(forWritingTo: url)
        defer { try? out.close() }
        var len = UInt32(head.count).littleEndian
        out.write("SOMNO001".data(using: .ascii)!)
        out.write(Data(bytes: &len, count: 4))
        out.write(head)
        for (s, d) in zip(sections, descr) {
            let inp = try FileHandle(forReadingFrom: s.file.url)
            defer { try? inp.close() }
            var remaining = d["bytes"] as? Int ?? 0
            while remaining > 0 {
                let chunk = inp.readData(ofLength: min(remaining, 4 << 20))
                if chunk.isEmpty { break }
                out.write(chunk)
                remaining -= chunk.count
            }
        }
    }
}
