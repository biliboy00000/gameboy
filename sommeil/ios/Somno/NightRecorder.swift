import Foundation
import SwiftUI
import UIKit

enum Placement: String, CaseIterable, Identifiable {
    case mattress = "Sur le matelas"
    case sonar = "Table de nuit · sonar"
    case depth = "Support · caméra TrueDepth"
    var id: String { rawValue }
    var detail: String {
        switch self {
        case .mattress: return "À plat sous le drap-housse ou l'oreiller, au niveau de la poitrine. Écran éteint possible."
        case .sonar: return "Haut-parleur vers vous, à moins d'1 m. Ultrasons 18–20,5 kHz. Écran éteint possible."
        case .depth: return "Écran face au lit, à 0,5–1 m, sur un support. Écran allumé (noir) toute la nuit."
        }
    }
}

struct NightFile: Identifiable, Hashable {
    let url: URL
    let start: Date
    var id: URL { url }
}

/// Orchestration d'une nuit : capteurs → fichiers bruts → fichier .somno final
/// (même format que l'export de l'app web, lisible par l'analyse Python).
@MainActor
final class NightRecorder: ObservableObject {
    @Published var recording = false
    @Published var startDate: Date?
    @Published var status = ""
    @Published var nights = [NightFile]()
    @Published var lastError: String?

    private var motion: MotionCapture?
    private var audio: AudioCapture?
    private var depth: DepthCapture?
    private var dir: URL?
    private var files = [String: StreamFile]()
    private var placement = Placement.mattress
    private var timer: Timer?

    static var nightsDir: URL {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("nuits")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    init() { refresh() }

    func refresh() {
        let urls = (try? FileManager.default.contentsOfDirectory(at: Self.nightsDir, includingPropertiesForKeys: nil)) ?? []
        nights = urls.filter { $0.pathExtension == "somno" }.compactMap { u in
            let ms = Double(u.deletingPathExtension().lastPathComponent) ?? 0
            return NightFile(url: u, start: Date(timeIntervalSince1970: ms / 1000))
        }.sorted { $0.start > $1.start }
    }

    func start(placement: Placement, sounds: Bool, clips: Bool) {
        lastError = nil
        self.placement = placement
        let now = Date()
        let id = String(Int(now.timeIntervalSince1970 * 1000))
        let d = Self.nightsDir.appendingPathComponent("tmp-" + id)
        do {
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            dir = d
            files = [:]
            let m = MotionCapture()
            files["motion"] = try StreamFile(url: d.appendingPathComponent("motion.f32"), columns: 6)
            m.start(sink: files["motion"]!)
            motion = m
            let useSonar = placement == .sonar
            if sounds || useSonar || placement == .mattress {
                // l'audio garde aussi l'app vivante écran éteint (mode d'arrière-plan « audio »)
                let a = AudioCapture()
                a.keepClips = clips && sounds
                files["audio"] = try StreamFile(url: d.appendingPathComponent("audio.f32"), columns: 8)
                if useSonar { files["sonar"] = try StreamFile(url: d.appendingPathComponent("sonar.f32"), columns: 2) }
                try a.start(sonar: useSonar, featSink: files["audio"]!, sonarSink: files["sonar"], clipDir: d)
                audio = a
            }
            if placement == .depth {
                let c = DepthCapture()
                files["depth"] = try StreamFile(url: d.appendingPathComponent("depth.f32"), columns: 16)
                try c.start(sink: files["depth"]!)
                depth = c
                UIApplication.shared.isIdleTimerDisabled = true
            }
            startDate = now
            recording = true
            timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.updateStatus() }
            }
        } catch {
            lastError = error.localizedDescription
            stopCaptures()
        }
    }

    private func updateStatus() {
        var parts = [String]()
        if let m = motion { parts.append("\(m.samples / 50 / 60) min de mouvement") }
        if let d = audio?.lastSonarDistance { parts.append(String(format: "sonar %.2f m", d)) }
        if let c = depth, c.frames > 0 { parts.append(String(format: "profondeur %.0f mm", c.lastCenterMM)) }
        status = parts.joined(separator: " · ")
    }

    private func stopCaptures() {
        motion?.stop()
        audio?.stop()
        depth?.stop()
        files.values.forEach { $0.close() }
        UIApplication.shared.isIdleTimerDisabled = false
        timer?.invalidate()
    }

    /// Termine la nuit et écrit le fichier .somno. → URL du fichier.
    func stop() -> URL? {
        stopCaptures()
        recording = false
        guard let dir, let start = startDate else { return nil }
        let id = String(Int(start.timeIntervalSince1970 * 1000))
        let out = Self.nightsDir.appendingPathComponent(id + ".somno")
        var sections = [SomnoWriter.Section]()
        if let f = files["motion"] { sections.append(.init(kind: "motion", file: f, rate: MotionCapture.fs,
            columns: ["ax", "ay", "az", "gx", "gy", "gz"])) }
        if let f = files["audio"] { sections.append(.init(kind: "audio", file: f, rate: 4,
            columns: ["db", "b50_150", "b150_500", "b500_1k", "b1k_2k", "b2k_4k", "b4k_8k", "peak_db"])) }
        if let f = files["sonar"], let a = audio { sections.append(.init(kind: "sonar", file: f, rate: a.sampleRate / 1024,
            columns: ["disp_mm", "act"], extra: ["epochs": a.sonarEpochs])) }
        if let f = files["depth"] { sections.append(.init(kind: "depth", file: f, rate: DepthCapture.rate,
            columns: (0..<16).map { "cell\($0)" })) }
        let meta: [String: Any] = [
            "id": id, "start": start.timeIntervalSince1970 * 1000, "end": Date().timeIntervalSince1970 * 1000,
            "fs": MotionCapture.fs, "audioRate": audio?.sampleRate ?? 48000,
            "opts": ["place": placement.rawValue, "sonar": placement == .sonar, "audio": audio != nil],
            "gaps": [], "ua": "Somno iOS (natif)",
            "motionStats": ["rateHz": 100, "stepAcc": NSNull(), "stepGyro": NSNull()],
        ]
        do {
            try SomnoWriter.write(to: out, meta: meta, sections: sections)
            // les extraits sonores (.wav) restent dans le dossier « nuits/<id> » (app Fichiers)
            let clipsDir = Self.nightsDir.appendingPathComponent(id)
            try? FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
            for u in (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
            where u.pathExtension == "wav" {
                try? FileManager.default.moveItem(at: u, to: clipsDir.appendingPathComponent(u.lastPathComponent))
            }
            try? FileManager.default.removeItem(at: dir)
        } catch {
            lastError = "Écriture impossible : \(error.localizedDescription)"
        }
        motion = nil; audio = nil; depth = nil; self.dir = nil
        refresh()
        return out
    }
}
