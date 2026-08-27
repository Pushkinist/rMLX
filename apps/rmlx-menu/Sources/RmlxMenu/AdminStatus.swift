import Foundation

struct AdminStatus: Decodable, Equatable, Sendable {
    var config: Config?
    var server: Server
    var models: [ModelEntry]
    var model: Model?
    var memory: Memory?
    var cache: Cache?
    var claim: Claim?

    struct Config: Decodable, Equatable, Sendable {
        var adminHost: String?
        var adminPort: Int?
        var serverHost: String?
        var serverPort: Int?

        private enum CodingKeys: String, CodingKey {
            case adminHost = "admin_host"
            case adminPort = "admin_port"
            case serverHost = "server_host"
            case serverPort = "server_port"
        }
    }

    struct Server: Decodable, Equatable, Sendable {
        var running: Bool
        var pid: Int?
        var port: Int?
        var healthy: Bool?
        var supervised: Bool?
        var uptimeSecs: Int?

        private enum CodingKeys: String, CodingKey {
            case running
            case pid
            case port
            case healthy
            case supervised
            case uptimeSecs = "uptime_secs"
        }
    }

    struct Model: Decodable, Equatable, Sendable {
        var id: String?
        var status: ModelStatus?
        var keepAliveSecs: Int?

        private enum CodingKeys: String, CodingKey {
            case id
            case status
            case keepAliveSecs = "keep_alive_secs"
        }
    }

    struct ModelEntry: Decodable, Equatable, Identifiable, Sendable {
        var id: String
        var loaded: Bool
    }

    struct Memory: Decodable, Equatable, Sendable {
        var rssBytes: Int64?
        var metalPeakAllocBytes: Int64?
        var kvCacheBytes: Int64?

        private enum CodingKeys: String, CodingKey {
            case rssBytes = "rss_bytes"
            case metalPeakAllocBytes = "metal_peak_alloc_bytes"
            case kvCacheBytes = "kv_cache_bytes"
        }
    }

    struct Cache: Decodable, Equatable, Sendable {
        var hits: Int?
        var misses: Int?
        var evictions: Int?
        var ssdHits: Int?
        var bytes: Int64?

        private enum CodingKeys: String, CodingKey {
            case hits
            case misses
            case evictions
            case ssdHits = "ssd_hits"
            case bytes
        }
    }

    struct Claim: Decodable, Equatable, Sendable {
        var held: Bool?
        var holderPid: Int?

        private enum CodingKeys: String, CodingKey {
            case held
            case holderPid = "holder_pid"
        }
    }
}

enum ModelStatus: String, Decodable, Equatable, Sendable {
    case loaded
    case loading
    case unloading
    case unloaded
    case error
    case unknown

    init(from decoder: Decoder) throws {
        let rawValue = try decoder.singleValueContainer().decode(String.self)
        self = ModelStatus(rawValue: rawValue) ?? .unknown
    }
}
