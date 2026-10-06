import AVFoundation
import JavaScriptCore

/// Micro (et haut-parleur pour le sonar) avec AVAudioEngine. Le traitement est délégué
/// au script web/core/somno-core.js (le MÊME code que l'app web, validé contre Python),
/// exécuté dans JavaScriptCore : caractéristiques sonores 4×/s, extraits, sonar FMCW.
/// L'app reste active écran éteint grâce au mode d'arrière-plan « audio ».
final class AudioCapture {
    private let engine = AVAudioEngine()
    private let jsQueue = DispatchQueue(label: "somno.js")
    private var ctx: JSContext!
    private var core: JSValue!
    private var source: AVAudioSourceNode?
    private var featSink: StreamFile?
    private var sonarSink: StreamFile?
    private var clipDir: URL?
    private(set) var sampleRate: Double = 48000
    private(set) var sonarEpochs = [[String: Any]]()
    private(set) var lastSonarDistance: Double?
    private var background: Double?
    private var lastClip: Double = -1e9
    private var clipCount = 0
    var keepClips = true

    func start(sonar: Bool, featSink: StreamFile, sonarSink: StreamFile?, clipDir: URL) throws {
        self.featSink = featSink
        self.sonarSink = sonarSink
        self.clipDir = clipDir
        let session = AVAudioSession.sharedInstance()
        // .measurement : pas d'égalisation ni d'annulation d'écho (indispensable au sonar)
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .mixWithOthers])
        try session.setPreferredSampleRate(48000)
        try session.setActive(true)

        let input = engine.inputNode
        let fmt = input.outputFormat(forBus: 0)
        sampleRate = fmt.sampleRate

        guard let url = Bundle.main.url(forResource: "somno-core", withExtension: "js", subdirectory: "web/core"),
              let src = try? String(contentsOf: url, encoding: .utf8) else {
            throw NSError(domain: "Somno", code: 1, userInfo: [NSLocalizedDescriptionKey: "somno-core.js absent du bundle"])
        }
        ctx = JSContext()
        ctx.exceptionHandler = { _, e in print("JS:", e?.toString() ?? "?") }
        ctx.evaluateScript(src)
        core = ctx.objectForKeyedSubscript("SomnoCore")
        core.invokeMethod("init", withArguments: [["sonar": sonar, "sampleRate": sampleRate]])

        input.installTap(onBus: 0, bufferSize: 4800, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength)
            let copy = Array(UnsafeBufferPointer(start: ch, count: n))
            self.jsQueue.async { self.process(copy) }
        }

        if sonar {
            let chirp = Self.chirp(sampleRate: sampleRate)
            var pos = 0
            let gain: Float = 0.35
            let node = AVAudioSourceNode { _, _, frameCount, abl -> OSStatus in
                let list = UnsafeMutableAudioBufferListPointer(abl)
                for f in 0..<Int(frameCount) {
                    let v = gain * chirp[pos]
                    pos = (pos + 1) % chirp.count
                    for b in list { b.mData?.assumingMemoryBound(to: Float.self)[f] = v }
                }
                return noErr
            }
            let outFmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: outFmt)
            engine.mainMixerNode.outputVolume = 1
            source = node
        } else {
            engine.mainMixerNode.outputVolume = 0
        }
        engine.prepare()
        try engine.start()
    }

    /// Même chirp que l'AudioWorklet (18 → 20,5 kHz, 1024 échantillons, bords de Tukey).
    static func chirp(sampleRate fs: Double) -> [Float] {
        let n = 1024, f0 = 18000.0, f1 = 20500.0
        let T = Double(n) / fs
        let m = Int(0.08 * Double(n) / 2)
        return (0..<n).map { i in
            let t = Double(i) / fs
            var v = sin(2 * .pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t))
            if i < m { v *= 0.5 - 0.5 * cos(.pi * Double(i) / Double(m)) }
            if i >= n - m { v *= 0.5 - 0.5 * cos(.pi * Double(n - 1 - i) / Double(m)) }
            return Float(v)
        }
    }

    private func process(_ samples: [Float]) {
        let n = samples.count
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: max(1, n))
        ptr.initialize(from: samples, count: n)
        let ref = JSObjectMakeTypedArrayWithBytesNoCopy(
            ctx.jsGlobalContextRef, kJSTypedArrayTypeFloat32Array, ptr, n * 4,
            { bytes, _ in bytes?.assumingMemoryBound(to: Float.self).deallocate() }, nil, nil)
        let arr = JSValue(jsValueRef: ref, in: ctx)!
        core.invokeMethod("pushAudio", withArguments: [arr])
        drain()
    }

    private func drain() {
        guard let r = core.invokeMethod("drain", withArguments: [])?.toDictionary() else { return }
        if let feats = r["feats"] as? [Double], !feats.isEmpty {
            featSink?.appendMany(feats.map(Float.init))
            detectSoundEvents(feats)
        }
        if let eps = r["epochs"] as? [[String: Any]] {
            for e in eps {
                let disp = (e["disp"] as? [Double]) ?? []
                let act = (e["act"] as? [Double]) ?? []
                var rows = [Float]()
                rows.reserveCapacity(disp.count * 2)
                for i in 0..<disp.count { rows.append(Float(disp[i])); rows.append(Float(i < act.count ? act[i] : 0)) }
                sonarSink?.appendMany(rows)
                sonarEpochs.append(["seq": e["epoch"] ?? 0, "bin": e["bin"] ?? 0, "distM": e["distM"] ?? 0])
                lastSonarDistance = e["distM"] as? Double
            }
        }
        if let clips = r["clips"] as? [[String: Any]] {
            for c in clips { saveClip(c) }
        }
    }

    /// Bruit marquant (> fond + 15 dB) → extrait de 10 s demandé 5 s plus tard.
    private func detectSoundEvents(_ feats: [Double]) {
        guard keepClips else { return }
        var i = 0
        while i + 8 <= feats.count {
            let db = feats[i]
            let bg = background ?? db
            background = bg + (db < bg ? 0.05 : 0.002) * (db - bg)
            let t = core.objectForKeyedSubscript("time").toDouble()
            if db > bg + 15, t - lastClip > 30, clipCount < 60 {
                lastClip = t
                clipCount += 1
                jsQueue.asyncAfter(deadline: .now() + 5) { [weak self] in
                    self?.core.invokeMethod("requestClip", withArguments: [t])
                    self?.drain()
                }
            }
            i += 8
        }
    }

    private func saveClip(_ c: [String: Any]) {
        guard let dir = clipDir, let data = c["data"] as? [Double], let rate = c["rate"] as? Double else { return }
        let t = (c["t"] as? Double) ?? 0
        let url = dir.appendingPathComponent(String(format: "son-%06.0f.wav", t))
        var pcm = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; pcm.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; pcm.append(Data(bytes: &x, count: 2)) }
        let peak = max(1e-6, data.map { abs($0) }.max() ?? 1)
        let g = min(20.0, 0.8 / peak)
        pcm.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + data.count * 2))
        pcm.append("WAVEfmt ".data(using: .ascii)!); u32(16); u16(1); u16(1)
        u32(UInt32(rate)); u32(UInt32(rate) * 2); u16(2); u16(16)
        pcm.append("data".data(using: .ascii)!); u32(UInt32(data.count * 2))
        for v in data { u16(UInt16(bitPattern: Int16(max(-32767, min(32767, v * g * 32767))))) }
        try? pcm.write(to: url)
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        jsQueue.sync {
            core?.invokeMethod("finish", withArguments: [])
            drain()
        }
        featSink?.flush()
        sonarSink?.flush()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
