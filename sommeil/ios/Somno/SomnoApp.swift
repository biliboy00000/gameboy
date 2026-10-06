import SwiftUI

@main
struct SomnoApp: App {
    @StateObject private var recorder = NightRecorder()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(recorder)
                .preferredColorScheme(.dark)
        }
    }
}
