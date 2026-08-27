import Foundation

struct AdminClient: Sendable {
    enum ClientError: Error, LocalizedError, Equatable {
        case invalidBaseURL(String)
        case nonLocalBaseURL(String)
        case invalidResponse
        case httpStatus(Int)

        var errorDescription: String? {
            switch self {
            case .invalidBaseURL(let value):
                "Invalid admin URL: \(value)"
            case .nonLocalBaseURL(let value):
                "Admin URL must be local: \(value)"
            case .invalidResponse:
                "The admin API returned an invalid response."
            case .httpStatus(let statusCode):
                "The admin API returned HTTP \(statusCode)."
            }
        }
    }

    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(
        session: URLSession = .shared,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) {
        self.session = session
        self.decoder = decoder
        self.encoder = encoder
    }

    func status(baseURLString: String) async throws -> AdminStatus {
        let baseURL = try validatedLocalBaseURL(baseURLString)

        let url = baseURL.appending(path: "admin/status")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 2

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ClientError.invalidResponse
        }
        guard 200..<300 ~= httpResponse.statusCode else {
            throw ClientError.httpStatus(httpResponse.statusCode)
        }

        return try decoder.decode(AdminStatus.self, from: data)
    }

    func loadModel(baseURLString: String, modelID: String, keepAliveSeconds: Int? = nil) async throws {
        let body = ModelLoadRequest(keepAlive: keepAliveSeconds)
        try await post(
            baseURLString: baseURLString,
            pathComponents: ["admin", "models", modelID, "load"],
            body: body
        )
    }

    func unloadModel(baseURLString: String, modelID: String) async throws {
        try await post(
            baseURLString: baseURLString,
            pathComponents: ["admin", "models", modelID, "unload"]
        )
    }

    func startServer(baseURLString: String) async throws {
        try await post(baseURLString: baseURLString, pathComponents: ["admin", "server", "start"])
    }

    func stopServer(baseURLString: String) async throws {
        try await post(baseURLString: baseURLString, pathComponents: ["admin", "server", "stop"])
    }

    func restartServer(baseURLString: String) async throws {
        try await post(baseURLString: baseURLString, pathComponents: ["admin", "server", "restart"])
    }

    private func post(baseURLString: String, pathComponents: [String]) async throws {
        let emptyBody: EmptyBody? = nil
        try await post(baseURLString: baseURLString, pathComponents: pathComponents, body: emptyBody)
    }

    private func post<Body: Encodable>(
        baseURLString: String,
        pathComponents: [String],
        body: Body?
    ) async throws {
        let baseURL = try validatedLocalBaseURL(baseURLString)

        let url = pathComponents.reduce(baseURL) { partialURL, component in
            partialURL.appending(path: component)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 10

        if let body {
            request.httpBody = try encoder.encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (_, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ClientError.invalidResponse
        }
        guard 200..<300 ~= httpResponse.statusCode else {
            throw ClientError.httpStatus(httpResponse.statusCode)
        }
    }

    private func validatedLocalBaseURL(_ value: String) throws -> URL {
        guard let url = URL(string: value),
              let scheme = url.scheme,
              let host = url.host,
              scheme == "http" || scheme == "https" else {
            throw ClientError.invalidBaseURL(value)
        }

        guard Self.isLoopbackHost(host) else {
            throw ClientError.nonLocalBaseURL(value)
        }

        return url
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: CharacterSet(charactersIn: "[]")).lowercased()
        return normalized == "localhost"
            || normalized == "::1"
            || normalized.hasPrefix("127.")
    }
}

private struct EmptyBody: Encodable {}

private struct ModelLoadRequest: Encodable {
    var keepAlive: Int?

    private enum CodingKeys: String, CodingKey {
        case keepAlive = "keep_alive"
    }
}
