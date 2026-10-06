import CoreMotion
import Foundation

/// Accéléromètre (gravité incluse, m/s²) + gyroscope (rad/s) à 100 Hz, ré-échantillonnés
/// sur une grille exacte de 50 Hz après moyenne de 2 points (bruit ÷ √2) — même format
/// que l'app web : [ax, ay, az, gx, gy, gz] par ligne.
final class MotionCapture {
    static let fs = 50.0
    private let manager = CMMotionManager()
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInitiated
        return q
    }()
    private var sink: StreamFile?
    private var prev: (t: Double, v: [Double])?
    private var last: [Double]?
    private var gridT: Double = 0
    private(set) var samples = 0
    private(set) var gaps = 0

    var isAvailable: Bool { manager.isDeviceMotionAvailable }

    func start(sink: StreamFile) {
        self.sink = sink
        manager.deviceMotionUpdateInterval = 1.0 / 100.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] dm, _ in
            guard let self, let dm else { return }
            let g = 9.80665
            let raw = [
                (dm.gravity.x + dm.userAcceleration.x) * g,
                (dm.gravity.y + dm.userAcceleration.y) * g,
                (dm.gravity.z + dm.userAcceleration.z) * g,
                dm.rotationRate.x, dm.rotationRate.y, dm.rotationRate.z,
            ]
            // moyenne glissante de 2 échantillons (anti-repliement simple avant 50 Hz)
            let v = self.last.map { l in zip(l, raw).map { ($0 + $1) / 2 } } ?? raw
            self.last = raw
            self.push(t: dm.timestamp, v: v)
        }
    }

    private func push(t: Double, v: [Double]) {
        let step = 1.0 / Self.fs
        guard let p = prev else {
            prev = (t, v)
            gridT = t
            return
        }
        if t - p.t > 1.0 { gaps += 1 }
        while gridT <= t {
            let f = (t - p.t) > 1.0 ? 1.0 : (gridT - p.t) / max(1e-9, t - p.t)
            let row = zip(p.v, v).map { Float($0 + ($1 - $0) * f) }
            sink?.append(row)
            samples += 1
            gridT += step
        }
        prev = (t, v)
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        queue.waitUntilAllOperationsAreFinished()
        sink?.flush()
    }
}
