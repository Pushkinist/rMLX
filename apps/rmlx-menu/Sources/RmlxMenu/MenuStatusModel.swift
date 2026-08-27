import Foundation

@MainActor
final class MenuStatusModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case loading
        case loaded(AdminStatus)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var lastRefresh: Date?
    @Published private(set) var actionMessage: String?
    @Published private(set) var isPerformingAction = false

    private let client: AdminClient
    private var pollingTask: Task<Void, Never>?

    init(client: AdminClient = AdminClient()) {
        self.client = client
    }

    deinit {
        pollingTask?.cancel()
    }

    var menuBarTitle: String {
        switch phase {
        case .idle, .loading:
            return "rMLX"
        case .failed:
            return "rMLX: Offline"
        case .loaded(let status):
            guard status.server.running else {
                return "rMLX: Stopped"
            }

            let modelName = status.model?.id ?? "Idle"
            if let memoryBytes = status.memory?.metalPeakAllocBytes ?? status.memory?.rssBytes {
                return "rMLX: \(modelName)  \(StatusFormatter.bytes(memoryBytes))"
            }
            return "rMLX: \(modelName)"
        }
    }

    var symbolName: String {
        switch phase {
        case .idle, .loading:
            return "circle.dashed"
        case .failed:
            return "exclamationmark.circle.fill"
        case .loaded(let status):
            if status.server.healthy == false {
                return "exclamationmark.circle.fill"
            }
            if status.model?.status == .loading || status.model?.status == .unloading {
                return "arrow.triangle.2.circlepath.circle.fill"
            }
            if status.model?.status == .loaded {
                return "circle.fill"
            }
            return status.server.running ? "circle" : "circle.slash"
        }
    }

    func startPolling(baseURLString: String) {
        pollingTask?.cancel()
        pollingTask = Task { [weak self] in
            await self?.poll(baseURLString: baseURLString)
        }
    }

    func refresh(baseURLString: String) async {
        actionMessage = nil
        phase = .loading
        await loadStatus(baseURLString: baseURLString)
    }

    func loadSelectedModel(baseURLString: String) async {
        guard let modelID = currentStatus?.model?.id else {
            showPlaceholder("No model is selected.")
            return
        }

        await loadModel(baseURLString: baseURLString, modelID: modelID)
    }

    func loadModel(baseURLString: String, modelID: String) async {
        await performAction("Loading \(modelID)...", baseURLString: baseURLString) {
            try await client.loadModel(baseURLString: baseURLString, modelID: modelID)
        }
    }

    func unloadCurrentModel(baseURLString: String) async {
        guard let modelID = currentStatus?.model?.id else {
            showPlaceholder("No model is currently loaded.")
            return
        }

        await unloadModel(baseURLString: baseURLString, modelID: modelID)
    }

    func unloadModel(baseURLString: String, modelID: String) async {
        await performAction("Unloading \(modelID)...", baseURLString: baseURLString) {
            try await client.unloadModel(baseURLString: baseURLString, modelID: modelID)
        }
    }

    func startServer(baseURLString: String) async {
        await performAction("Starting server...", baseURLString: baseURLString) {
            try await client.startServer(baseURLString: baseURLString)
        }
    }

    func stopServer(baseURLString: String) async {
        await performAction("Stopping server...", baseURLString: baseURLString) {
            try await client.stopServer(baseURLString: baseURLString)
        }
    }

    func restartServer(baseURLString: String) async {
        await performAction("Restarting server...", baseURLString: baseURLString) {
            try await client.restartServer(baseURLString: baseURLString)
        }
    }

    func showPlaceholder(_ message: String = "This action is not available yet.") {
        actionMessage = message
    }

    var currentStatus: AdminStatus? {
        guard case .loaded(let status) = phase else {
            return nil
        }
        return status
    }

    private func poll(baseURLString: String) async {
        await loadStatus(baseURLString: baseURLString)

        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(5))
            } catch {
                return
            }
            await loadStatus(baseURLString: baseURLString)
        }
    }

    private func loadStatus(baseURLString: String) async {
        do {
            let status = try await client.status(baseURLString: baseURLString)
            phase = .loaded(status)
            lastRefresh = Date()
        } catch {
            phase = .failed(error.localizedDescription)
            lastRefresh = Date()
        }
    }

    private func performAction(
        _ pendingMessage: String,
        baseURLString: String,
        action: () async throws -> Void
    ) async {
        guard !isPerformingAction else {
            return
        }

        isPerformingAction = true
        actionMessage = pendingMessage
        defer {
            isPerformingAction = false
        }

        do {
            try await action()
            actionMessage = "Action sent."
        } catch AdminClient.ClientError.httpStatus(404),
                AdminClient.ClientError.httpStatus(501) {
            actionMessage = "This action is not available in the daemon yet."
        } catch {
            actionMessage = error.localizedDescription
        }

        await loadStatus(baseURLString: baseURLString)
    }
}
