import SwiftUI
import UniformTypeIdentifiers
import WebKit

/// Affiche l'application web embarquée (analyse + rapport) dans un WKWebView, en lui
/// servant les fichiers du bundle et la nuit à analyser via le schéma « somno:// ».
/// Le rapport calculé en JavaScript revient ici → Apple Santé.
struct ReportView: UIViewRepresentable {
    let night: URL
    var onReport: ([String: Any]) -> Void = { _ in }

    func makeCoordinator() -> Coordinator { Coordinator(onReport: onReport) }

    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.setURLSchemeHandler(context.coordinator.scheme, forURLScheme: "somno")
        cfg.userContentController.add(context.coordinator, name: "somno")
        cfg.websiteDataStore = .default()
        let web = WKWebView(frame: .zero, configuration: cfg)
        web.isOpaque = false
        web.backgroundColor = .black
        let file = night.lastPathComponent
        context.coordinator.scheme.nightURL = night
        web.load(URLRequest(url: URL(string: "somno://app/index.html?import=somno://app/nuit/\(file)")!))
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKScriptMessageHandler {
        let scheme = SchemeHandler()
        let onReport: ([String: Any]) -> Void
        init(onReport: @escaping ([String: Any]) -> Void) { self.onReport = onReport }

        func userContentController(_ c: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any], body["type"] as? String == "report",
                  let rep = body["report"] as? [String: Any] else { return }
            onReport(rep)
        }
    }

    final class SchemeHandler: NSObject, WKURLSchemeHandler {
        var nightURL: URL?
        private let webRoot = Bundle.main.url(forResource: "web", withExtension: nil)!

        func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
            guard let url = task.request.url else { return }
            var path = url.path
            if path.isEmpty || path == "/" { path = "/index.html" }
            let fileURL: URL
            if path.hasPrefix("/nuit/"), let n = nightURL {
                fileURL = n
            } else {
                fileURL = webRoot.appendingPathComponent(String(path.dropFirst()))
            }
            guard let data = try? Data(contentsOf: fileURL) else {
                task.didReceive(HTTPURLResponse(url: url, statusCode: 404, httpVersion: nil, headerFields: nil)!)
                task.didFinish()
                return
            }
            let mime: String
            switch fileURL.pathExtension {
            case "js", "mjs": mime = "text/javascript"
            case "html": mime = "text/html"
            case "css": mime = "text/css"
            case "json", "webmanifest": mime = "application/json"
            case "png": mime = "image/png"
            case "mp4": mime = "video/mp4"
            default: mime = "application/octet-stream"
            }
            task.didReceive(HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                            headerFields: ["Content-Type": mime, "Content-Length": "\(data.count)"])!)
            task.didReceive(data)
            task.didFinish()
        }

        func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
    }
}
