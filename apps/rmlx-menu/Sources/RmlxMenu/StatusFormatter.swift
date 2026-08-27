import Foundation

enum StatusFormatter {
    static func bytes(_ value: Int64?) -> String {
        guard let value else {
            return "--"
        }

        return ByteCountFormatter.string(
            fromByteCount: value,
            countStyle: .memory
        )
    }

    static func uptime(_ seconds: Int?) -> String {
        guard let seconds, seconds > 0 else {
            return "--"
        }

        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = seconds >= 3_600 ? [.hour, .minute] : [.minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: TimeInterval(seconds)) ?? "--"
    }

    static func keepAlive(_ seconds: Int?) -> String {
        guard let seconds else {
            return "--"
        }

        if seconds < 0 {
            return "Keep loaded"
        }
        if seconds == 0 {
            return "After next request"
        }
        return uptime(seconds)
    }
}
