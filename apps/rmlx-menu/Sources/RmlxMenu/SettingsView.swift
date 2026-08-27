import SwiftUI

struct SettingsView: View {
    @Binding var adminBaseURL: String
    @State private var modelRegistryPath = "Managed by rmlxd"
    @State private var defaultIdleTimeout = "Managed by rmlxd"
    @State private var promptCacheRAMGiB = 2
    @State private var ssdKVCacheGiB = 20
    @State private var ssdProjectNamespace = "Managed by rmlxd"
    @State private var maxLoadedModels = 1
    @State private var defaultKVQuant = "Managed by rmlxd"
    @State private var launchAtLogin = false
    @State private var logLocation = "Managed by rmlxd"

    var body: some View {
        Form {
            Section("Daemon") {
                TextField("Admin API URL", text: $adminBaseURL)
                TextField("Model registry path", text: $modelRegistryPath)
                    .disabled(true)
                Toggle("Launch at login", isOn: $launchAtLogin)
                    .disabled(true)
            }

            Section("Server") {
                Picker("Default idle timeout", selection: $defaultIdleTimeout) {
                    Text("Managed by rmlxd").tag("Managed by rmlxd")
                }
                .disabled(true)
                Stepper("Max loaded models: \(maxLoadedModels)", value: $maxLoadedModels, in: 1...8)
                    .disabled(true)
                TextField("Default KV quant", text: $defaultKVQuant)
                    .disabled(true)
            }

            Section("Cache") {
                Stepper("Prompt cache RAM: \(promptCacheRAMGiB) GiB", value: $promptCacheRAMGiB, in: 0...256)
                    .disabled(true)
                Stepper("SSD KV cache: \(ssdKVCacheGiB) GiB", value: $ssdKVCacheGiB, in: 0...4096)
                    .disabled(true)
                TextField("SSD project namespace", text: $ssdProjectNamespace)
                    .disabled(true)
            }

            Section("Logs") {
                TextField("Log location", text: $logLocation)
                    .disabled(true)
            }

            Section {
                Button("Restart Server") {}
                    .disabled(true)
            } footer: {
                Text("Settings are placeholders until rmlxd exposes config updates.")
            }
        }
        .formStyle(.grouped)
        .padding(20)
        .frame(width: 520)
    }
}
