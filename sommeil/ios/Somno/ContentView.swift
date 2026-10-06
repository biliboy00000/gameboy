import SwiftUI

struct ContentView: View {
    @EnvironmentObject var rec: NightRecorder
    @State private var placement = Placement.mattress
    @State private var sounds = true
    @State private var clips = true
    @State private var report: NightFile?
    @State private var healthMsg: String?

    var body: some View {
        if rec.recording {
            NightScreen()
        } else {
            NavigationStack {
                List {
                    Section("Placement de l'iPhone") {
                        Picker("Placement", selection: $placement) {
                            ForEach(Placement.allCases) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.inline)
                        .labelsHidden()
                        Text(placement.detail).font(.footnote).foregroundStyle(.secondary)
                    }
                    Section("Options") {
                        Toggle("Sons de la nuit (ronflements)", isOn: $sounds)
                        Toggle("Garder de courts extraits", isOn: $clips).disabled(!sounds)
                    }
                    Section {
                        Button {
                            rec.start(placement: placement, sounds: sounds, clips: clips)
                        } label: {
                            Label("Commencer la nuit", systemImage: "moon.zzz.fill").font(.headline)
                        }
                        if let e = rec.lastError { Text(e).foregroundStyle(.red).font(.footnote) }
                    } footer: {
                        Text("Branchez le chargeur. Le mode « sur le matelas » et le sonar fonctionnent écran éteint.")
                    }
                    Section("Nuits") {
                        if rec.nights.isEmpty { Text("Aucune nuit pour l'instant.").foregroundStyle(.secondary) }
                        ForEach(rec.nights) { n in
                            HStack {
                                Button {
                                    report = n
                                } label: {
                                    VStack(alignment: .leading) {
                                        Text(n.start.formatted(date: .complete, time: .omitted))
                                        Text("début \(n.start.formatted(date: .omitted, time: .shortened))")
                                            .font(.footnote).foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                ShareLink(item: n.url) { Image(systemName: "square.and.arrow.up") }
                                    .buttonStyle(.borderless)
                            }
                        }
                        .onDelete { idx in
                            for i in idx { try? FileManager.default.removeItem(at: rec.nights[i].url) }
                            rec.refresh()
                        }
                    }
                    if let m = healthMsg { Section { Text(m).font(.footnote) } }
                }
                .navigationTitle("☾ Somno")
                .sheet(item: $report) { n in
                    ReportView(night: n.url) { rep in
                        guard let hyp = rep["hyp"] as? [Int], let start = rep["start"] as? Double else { return }
                        Task {
                            do {
                                try await HealthExport.save(start: Date(timeIntervalSince1970: start / 1000), hyp: hyp)
                                healthMsg = "Phases de sommeil ajoutées à Apple Santé."
                            } catch {
                                healthMsg = "Apple Santé : \(error.localizedDescription)"
                            }
                        }
                    }
                    .ignoresSafeArea()
                }
            }
        }
    }
}

/// Écran de nuit : noir (OLED éteint), horloge rouge très sombre, arrêt par appui long.
struct NightScreen: View {
    @EnvironmentObject var rec: NightRecorder
    @State private var showControls = false

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            ZStack {
                Color.black.ignoresSafeArea()
                VStack(spacing: 6) {
                    Text(ctx.date.formatted(date: .omitted, time: .shortened))
                        .font(.system(size: 76, weight: .thin, design: .rounded))
                        .foregroundStyle(Color(red: 0.23, green: 0.11, blue: 0.11))
                    if let s = rec.startDate {
                        Text(Duration.seconds(ctx.date.timeIntervalSince(s)).formatted(.time(pattern: .hourMinute)))
                            .foregroundStyle(Color(white: 0.17))
                    }
                    Text(rec.status).font(.footnote).foregroundStyle(Color(white: 0.18))
                }
                VStack {
                    Spacer()
                    Text("Maintenir pour terminer la nuit")
                        .padding().frame(maxWidth: .infinity)
                        .background(Color(white: 0.05), in: RoundedRectangle(cornerRadius: 14))
                        .foregroundStyle(Color(white: 0.5))
                        .opacity(showControls ? 1 : 0.12)
                        .onLongPressGesture(minimumDuration: 1.5) { _ = rec.stop() }
                        .padding()
                }
            }
            .onTapGesture {
                showControls = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 5) { showControls = false }
            }
        }
        .statusBarHidden()
    }
}
