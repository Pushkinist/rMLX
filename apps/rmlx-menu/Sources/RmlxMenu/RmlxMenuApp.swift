import AppKit
import SwiftUI

@main
struct RmlxMenuApp: App {
    @AppStorage("adminBaseURL") private var adminBaseURL = "http://127.0.0.1:6276"
    @StateObject private var statusModel = MenuStatusModel()

    var body: some Scene {
        MenuBarExtra {
            RmlxMenuView(
                model: statusModel,
                adminBaseURL: adminBaseURL
            )
            .task(id: adminBaseURL) {
                statusModel.startPolling(baseURLString: adminBaseURL)
            }
        } label: {
            Label(statusModel.menuBarTitle, systemImage: statusModel.symbolName)
        }
        .menuBarExtraStyle(.menu)

        Settings {
            SettingsView(adminBaseURL: $adminBaseURL)
        }
    }
}

struct RmlxMenuView: View {
    @ObservedObject var model: MenuStatusModel
    var adminBaseURL: String

    var body: some View {
        switch model.phase {
        case .idle, .loading:
            ProgressView("Checking rMLX...")
            Divider()
            commonRows
        case .failed(let message):
            Section("rMLX") {
                Text("Status: Unavailable")
                Text(message)
                    .lineLimit(2)
            }
            ActionMessageRow(message: model.actionMessage)
            Divider()
            commonRows
        case .loaded(let status):
            StatusSection(status: status)
            Divider()
            ModelsSection(model: model, status: status, adminBaseURL: adminBaseURL)
            Divider()
            ActionsSection(model: model, status: status, adminBaseURL: adminBaseURL)
            ActionMessageRow(message: model.actionMessage)
            Divider()
            KeepAliveSection(status: status)
            Divider()
            CacheSection(model: model, status: status)
            Divider()
            ServerSection(model: model, status: status, adminBaseURL: adminBaseURL)
            Divider()
            commonRows
        }
    }

    @ViewBuilder
    private var commonRows: some View {
        Button("Refresh") {
            Task {
                await model.refresh(baseURLString: adminBaseURL)
            }
        }
        SettingsLink {
            Text("Settings...")
        }
        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
    }
}

private struct ActionMessageRow: View {
    var message: String?

    var body: some View {
        if let message {
            Text(message)
                .lineLimit(2)
        }
    }
}

private struct StatusSection: View {
    var status: AdminStatus

    var body: some View {
        Section("rMLX") {
            Text(serverStatus)
            Text("Model: \(status.model?.id ?? "None")")
            Text("Memory: \(StatusFormatter.bytes(status.memory?.metalPeakAllocBytes)) Metal peak, \(StatusFormatter.bytes(status.memory?.kvCacheBytes)) KV")
            Text("Cache: \(status.cache?.hits ?? 0) hits / \(status.cache?.misses ?? 0) misses, SSD \(status.cache?.ssdHits ?? 0) hits")
        }
    }

    private var serverStatus: String {
        guard status.server.running else {
            return "Status: Stopped"
        }

        if let port = status.server.port {
            return "Status: Running on :\(port)"
        }
        return "Status: Running"
    }
}

private struct ModelsSection: View {
    @ObservedObject var model: MenuStatusModel
    var status: AdminStatus
    var adminBaseURL: String

    var body: some View {
        Section("Models") {
            if status.models.isEmpty {
                Button("No registry models") {}
                    .disabled(true)
            } else {
                ForEach(status.models) { item in
                    Button(modelRowTitle(item)) {
                        Task {
                            if item.loaded {
                                await model.unloadModel(baseURLString: adminBaseURL, modelID: item.id)
                            } else {
                                await model.loadModel(baseURLString: adminBaseURL, modelID: item.id)
                            }
                        }
                    }
                    .disabled(model.isPerformingAction)
                }
            }
        }
    }

    private func modelRowTitle(_ item: AdminStatus.ModelEntry) -> String {
        let marker = item.loaded ? "* " : "  "
        let state = item.loaded ? "Loaded" : "Unloaded"
        return "\(marker)\(item.id)        \(state)"
    }
}

private struct ActionsSection: View {
    @ObservedObject var model: MenuStatusModel
    var status: AdminStatus
    var adminBaseURL: String

    var body: some View {
        Section("Actions") {
            Button("Load From Models List") {
                model.showPlaceholder("Choose a model from the Models section.")
            }
            .disabled(true)

            Button("Unload Current Model") {
                Task {
                    await model.unloadCurrentModel(baseURLString: adminBaseURL)
                }
            }
            .disabled(!canUnloadCurrentModel)

            Button("Start Server") {
                Task {
                    await model.startServer(baseURLString: adminBaseURL)
                }
            }
            .disabled(!canStartServer)

            Button("Restart Server") {
                Task {
                    await model.restartServer(baseURLString: adminBaseURL)
                }
            }
            .disabled(model.isPerformingAction)

            Button("Stop Server") {
                Task {
                    await model.stopServer(baseURLString: adminBaseURL)
                }
            }
            .disabled(!canStopServer)
        }
    }

    private var canUnloadCurrentModel: Bool {
        guard !model.isPerformingAction, status.model?.id != nil else {
            return false
        }
        return status.model?.status == .loaded || status.model?.status == .loading
    }

    private var canStartServer: Bool {
        !model.isPerformingAction && !status.server.running
    }

    private var canStopServer: Bool {
        !model.isPerformingAction && status.server.running && status.server.supervised == true
    }
}

private struct KeepAliveSection: View {
    var status: AdminStatus

    var body: some View {
        Section("Keep Alive") {
            Text("Current: \(StatusFormatter.keepAlive(status.model?.keepAliveSecs))")
            Button("After Each Request") {}
                .disabled(true)
            Button("5 minutes") {}
                .disabled(true)
            Button("15 minutes") {}
                .disabled(true)
            Button("1 hour") {}
                .disabled(true)
            Button("Keep Loaded") {}
                .disabled(true)
        }
    }
}

private struct CacheSection: View {
    @ObservedObject var model: MenuStatusModel
    var status: AdminStatus

    var body: some View {
        Section("Cache") {
            Text("Prompt/KV: \(StatusFormatter.bytes(status.cache?.bytes))")
            Text("Evictions: \(status.cache?.evictions ?? 0)")
            Button("Clear RAM Prompt Cache") {
                model.showPlaceholder()
            }
            .disabled(model.isPerformingAction)
            Button("Open Cache Folder") {
                model.showPlaceholder()
            }
            .disabled(model.isPerformingAction)
        }
    }
}

private struct ServerSection: View {
    @ObservedObject var model: MenuStatusModel
    var status: AdminStatus
    var adminBaseURL: String

    var body: some View {
        Section("Server") {
            Text(portTitle)
            Text("Uptime: \(StatusFormatter.uptime(status.server.uptimeSecs))")
            Text(claimTitle)
            Button("Copy OpenAI Base URL") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(openAIBaseURL, forType: .string)
            }
            Button("Open Logs") {
                model.showPlaceholder()
            }
            .disabled(model.isPerformingAction)
            Button("Open Config") {
                model.showPlaceholder()
            }
            .disabled(model.isPerformingAction)
        }
    }

    private var portTitle: String {
        guard let port = status.server.port else {
            return "Port: --"
        }
        return "Port: \(port)"
    }

    private var claimTitle: String {
        guard status.claim?.held == true else {
            return "Claim: Not held"
        }
        if let holderPid = status.claim?.holderPid {
            return "Claim: Held by PID \(holderPid)"
        }
        return "Claim: Held"
    }

    private var openAIBaseURL: String {
        let host = status.config?.serverHost ?? "127.0.0.1"
        let port = status.config?.serverPort ?? status.server.port

        guard let port else {
            return "http://\(host)/v1"
        }
        return "http://\(host):\(port)/v1"
    }
}
