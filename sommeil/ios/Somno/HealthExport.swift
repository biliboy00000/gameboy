import Foundation
import HealthKit

/// Enregistre l'hypnogramme estimé dans Apple Santé (Sommeil : Éveillé, Principal,
/// Profond, Paradoxal), comme le ferait une Apple Watch.
enum HealthExport {
    static let store = HKHealthStore()

    static func save(start: Date, hyp: [Int], epoch: TimeInterval = 30) async throws {
        guard HKHealthStore.isHealthDataAvailable(), !hyp.isEmpty else { return }
        let type = HKCategoryType(.sleepAnalysis)
        try await store.requestAuthorization(toShare: [type], read: [type])
        let source: [String: Any] = [HKMetadataKeyWasUserEntered: false]
        var samples = [HKCategorySample]()
        let end = start.addingTimeInterval(Double(hyp.count) * epoch)
        samples.append(HKCategorySample(type: type, value: HKCategoryValueSleepAnalysis.inBed.rawValue,
                                        start: start, end: end, metadata: source))
        // période de sommeil : du premier au dernier époque endormie
        guard let first = hyp.firstIndex(where: { $0 > 0 }), let last = hyp.lastIndex(where: { $0 > 0 }) else {
            try await store.save(samples)
            return
        }
        var i = first
        while i <= last {
            var j = i
            while j + 1 <= last && hyp[j + 1] == hyp[i] { j += 1 }
            let value: HKCategoryValueSleepAnalysis
            switch hyp[i] {
            case 1: value = .asleepCore
            case 2: value = .asleepDeep
            case 3: value = .asleepREM
            default: value = .awake
            }
            samples.append(HKCategorySample(type: type, value: value.rawValue,
                                            start: start.addingTimeInterval(Double(i) * epoch),
                                            end: start.addingTimeInterval(Double(j + 1) * epoch), metadata: source))
            i = j + 1
        }
        try await store.save(samples)
    }
}
