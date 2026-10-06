import AVFoundation

/// Mode expérimental : caméra de profondeur TrueDepth (projecteur infrarouge, fonctionne
/// dans le noir complet). iPhone sur un support, écran face au lit, à 0,5–1 m.
/// On enregistre la profondeur moyenne (mm) de 4×4 cellules couvrant le centre de l'image,
/// 15 fois par seconde ; l'analyse choisit la cellule « respiratoire » à chaque époque.
/// La caméra ne tourne qu'au premier plan : l'écran reste allumé (luminosité minimale).
final class DepthCapture: NSObject, AVCaptureDepthDataOutputDelegate {
    static let grid = 4
    static let rate = 15.0
    private let session = AVCaptureSession()
    private let output = AVCaptureDepthDataOutput()
    private let queue = DispatchQueue(label: "somno.depth")
    private var sink: StreamFile?
    private var acc = [Float](repeating: 0, count: 16)
    private var accN = 0
    private(set) var frames = 0
    private(set) var lastCenterMM: Float = 0

    func start(sink: StreamFile) throws {
        self.sink = sink
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            throw NSError(domain: "Somno", code: 2, userInfo: [NSLocalizedDescriptionKey: "Caméra TrueDepth indisponible"])
        }
        let input = try AVCaptureDeviceInput(device: device)
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw NSError(domain: "Somno", code: 3, userInfo: [NSLocalizedDescriptionKey: "Configuration caméra impossible"])
        }
        session.addInput(input)
        session.addOutput(output)
        output.isFilteringEnabled = true  // comble les trous, lissage spatial
        output.setDelegate(self, callbackQueue: queue)
        // format de profondeur flottant le plus fin disponible
        let depthFormats = device.activeFormat.supportedDepthDataFormats.filter {
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
                || CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
        }
        if let best = depthFormats.max(by: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width
                < CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
        }) {
            try device.lockForConfiguration()
            device.activeDepthDataFormat = best
            device.unlockForConfiguration()
        }
        session.commitConfiguration()
        queue.async { self.session.startRunning() }
    }

    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData,
                         timestamp: CMTime, connection: AVCaptureConnection) {
        let d = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let pb = d.depthDataMap
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let g = Self.grid
        // zone centrale : 60 % de la largeur et de la hauteur
        let x0 = Int(Double(w) * 0.2), x1 = Int(Double(w) * 0.8)
        let y0 = Int(Double(h) * 0.2), y1 = Int(Double(h) * 0.8)
        var sums = [Double](repeating: 0, count: g * g), counts = [Int](repeating: 0, count: g * g)
        for y in stride(from: y0, to: y1, by: 2) {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            let cy = min(g - 1, (y - y0) * g / max(1, y1 - y0))
            for x in stride(from: x0, to: x1, by: 2) {
                let v = row[x]
                guard v.isFinite, v > 0.1, v < 3 else { continue }
                let c = cy * g + min(g - 1, (x - x0) * g / max(1, x1 - x0))
                sums[c] += Double(v)
                counts[c] += 1
            }
        }
        for c in 0..<(g * g) {
            acc[c] += counts[c] > 0 ? Float(sums[c] / Double(counts[c]) * 1000) : 0
        }
        accN += 1
        frames += 1
        // ~30 i/s → moyenne de 2 trames → 15 i/s
        if accN == 2 {
            let row = acc.map { $0 / Float(accN) }
            sink?.append(row)
            lastCenterMM = row[g * g / 2]
            acc = [Float](repeating: 0, count: g * g)
            accN = 0
        }
    }

    func stop() {
        session.stopRunning()
        queue.sync {}
        sink?.flush()
    }
}
