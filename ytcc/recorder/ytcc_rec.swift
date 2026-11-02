// ytcc-rec: macOS meeting audio helper for `ytcc record`.
//
//   ytcc-rec detect
//       Print JSON listing the processes currently using a microphone.
//   ytcc-rec record --mic FILE --system FILE [--echo-cancel]
//       Record the microphone and all system audio output to two files until
//       SIGINT or SIGTERM. System audio comes from a Core Audio process tap
//       (macOS 14.4+), so the output device and volume stay untouched.
//
// Built on demand by ytcc/recorder.py with swiftc; Info.plist is embedded for
// the microphone and system audio permission prompts.

import AVFoundation
import CoreAudio
import Foundation

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func check(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw Failure("\(what) failed (OSStatus \(status))") }
}

// MARK: - Core Audio properties

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
}

func readValue<T: BitwiseCopyable>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T) throws -> T {
    var addr = address(selector)
    var size = UInt32(MemoryLayout<T>.size)
    var value = initial
    try check(AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value), "read property \(selector)")
    return value
}

func readArray(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> [AudioObjectID] {
    var addr = address(selector)
    var size: UInt32 = 0
    try check(AudioObjectGetPropertyDataSize(object, &addr, 0, nil, &size), "size of property \(selector)")
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    try check(AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &ids), "read property \(selector)")
    return ids
}

func readString(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var size = UInt32(MemoryLayout<CFString?>.size)
    var value: Unmanaged<CFString>?
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr, let value else { return nil }
    return value.takeRetainedValue() as String
}

// MARK: - detect

func detect() throws {
    var inputs: [[String: Any]] = []
    for process in try readArray(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList) {
        let running = (try? readValue(process, kAudioProcessPropertyIsRunningInput, UInt32(0))) ?? 0
        guard running != 0 else { continue }
        let pid = (try? readValue(process, kAudioProcessPropertyPID, pid_t(0))) ?? 0
        if pid == getpid() { continue }
        inputs.append([
            "pid": pid,
            "bundle_id": readString(process, kAudioProcessPropertyBundleID) ?? "",
        ])
    }
    let data = try JSONSerialization.data(withJSONObject: ["inputs": inputs], options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

// MARK: - record

/// Writes buffers to an ADTS AAC file (use a .aac path): its frames are self-
/// delimiting, so the file stays readable if the helper is killed mid-write.
///
/// Files are aligned to a shared start time: a stream that starts late, or
/// stops delivering buffers for a while (a tap with nothing playing), gets
/// silence written in its place, so both tracks share one timeline.
final class Writer {
    private var file: AVAudioFile?
    private let queue: DispatchQueue
    private let format: AVAudioFormat
    private let startHostTime: UInt64
    private var written: AVAudioFramePosition = 0

    init(url: URL, format: AVAudioFormat, startHostTime: UInt64, label: String) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVEncoderBitRatePerChannelKey: 48_000,
        ]
        file = try AVAudioFile(
            forWriting: url, settings: settings,
            commonFormat: format.commonFormat, interleaved: format.isInterleaved
        )
        self.format = format
        self.startHostTime = startHostTime
        queue = DispatchQueue(label: "ytcc-rec.\(label)")
    }

    /// Copies the buffer, so callers may reuse theirs once this returns.
    /// hostTime is when the buffer's first sample was captured.
    func write(_ buffer: AVAudioPCMBuffer, hostTime: UInt64) {
        guard let copy = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: buffer.frameLength) else { return }
        copy.frameLength = buffer.frameLength
        let src = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: buffer.audioBufferList))
        let dst = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        for (s, d) in zip(src, dst) {
            memcpy(d.mData, s.mData, Int(min(s.mDataByteSize, d.mDataByteSize)))
        }
        queue.async { [self] in
            do {
                if hostTime > startHostTime {
                    let due = AVAudioFramePosition(
                        AVAudioTime.seconds(forHostTime: hostTime - startHostTime) * format.sampleRate
                    )
                    // Fill gaps over 100 ms; smaller ones are clock jitter
                    if due - written > AVAudioFramePosition(format.sampleRate / 10) {
                        try writeSilence(frames: due - written)
                    }
                }
                try file?.write(from: copy)
                written += AVAudioFramePosition(copy.frameLength)
            } catch {
                log("write failed: \(error)")
            }
        }
    }

    private func writeSilence(frames: AVAudioFramePosition) throws {
        let chunk = AVAudioFrameCount(format.sampleRate)
        guard let silence = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunk) else { return }
        for b in UnsafeMutableAudioBufferListPointer(silence.mutableAudioBufferList) {
            memset(b.mData, 0, Int(b.mDataByteSize))
        }
        var left = frames
        while left > 0 {
            silence.frameLength = AVAudioFrameCount(min(left, AVAudioFramePosition(chunk)))
            try file?.write(from: silence)
            left -= AVAudioFramePosition(silence.frameLength)
            written += AVAudioFramePosition(silence.frameLength)
        }
    }

    func close() {
        queue.sync { file = nil }  // AVAudioFile finalizes on release
    }
}

final class MicRecorder {
    private let engine = AVAudioEngine()
    private let writer: Writer

    init(url: URL, startHostTime: UInt64, echoCancel: Bool) throws {
        let input = engine.inputNode
        if echoCancel {
            // Removes what the speakers play from the mic signal; keep other
            // apps' audio at full volume while it runs.
            try input.setVoiceProcessingEnabled(true)
            input.voiceProcessingOtherAudioDuckingConfiguration = .init(
                enableAdvancedDucking: false, duckingLevel: .min
            )
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw Failure("no microphone input available") }
        writer = try Writer(url: url, format: format, startHostTime: startHostTime, label: "mic")
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [writer] buffer, when in
            writer.write(buffer, hostTime: when.isHostTimeValid ? when.hostTime : mach_absolute_time())
        }
    }

    func start() throws { try engine.start() }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        writer.close()
    }
}

final class SystemRecorder {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var deviceID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private let writer: Writer
    private let format: AVAudioFormat

    init(url: URL, startHostTime: UInt64) throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.uuid = UUID()
        description.name = "ytcc system audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tapID), "create process tap")

        var asbd = try readValue(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription())
        guard let format = AVAudioFormat(streamDescription: &asbd) else { throw Failure("unsupported tap format") }
        self.format = format

        let output = try readValue(
            AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultSystemOutputDevice, AudioObjectID(0)
        )
        guard let outputUID = readString(output, kAudioDevicePropertyDeviceUID) else {
            throw Failure("no default output device")
        }
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "ytcc-rec",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [
                [kAudioSubTapDriftCompensationKey: true, kAudioSubTapUIDKey: description.uuid.uuidString],
            ],
        ]
        try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &deviceID), "create aggregate device")
        writer = try Writer(url: url, format: format, startHostTime: startHostTime, label: "system")
    }

    func start() throws {
        let queue = DispatchQueue(label: "ytcc-rec.tap", qos: .userInitiated)
        try check(
            AudioDeviceCreateIOProcIDWithBlock(&procID, deviceID, queue) { [writer, format] _, input, inputTime, _, _ in
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: input, deallocator: nil)
                else { return }
                let valid = inputTime.pointee.mFlags.contains(.hostTimeValid)
                writer.write(buffer, hostTime: valid ? inputTime.pointee.mHostTime : mach_absolute_time())
            },
            "create IO proc"
        )
        try check(AudioDeviceStart(deviceID, procID), "start aggregate device")
    }

    func stop() {
        if let procID {
            AudioDeviceStop(deviceID, procID)
            AudioDeviceDestroyIOProcID(deviceID, procID)
        }
        writer.close()
        teardown()
    }

    func teardown() {
        if deviceID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(deviceID) }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        deviceID = AudioObjectID(kAudioObjectUnknown)
        tapID = AudioObjectID(kAudioObjectUnknown)
    }
}

func record(mic: URL, system: URL, echoCancel: Bool) throws -> Never {
    let start = mach_absolute_time()
    let systemRecorder = try SystemRecorder(url: system, startHostTime: start)
    let micRecorder: MicRecorder
    do {
        micRecorder = try MicRecorder(url: mic, startHostTime: start, echoCancel: echoCancel)
        try systemRecorder.start()
        try micRecorder.start()
    } catch {
        systemRecorder.teardown()
        throw error
    }
    print("recording")  // `ytcc record start` waits for this line
    fflush(stdout)

    var sources: [DispatchSourceSignal] = []
    for sig in [SIGINT, SIGTERM] {
        signal(sig, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        source.setEventHandler {
            micRecorder.stop()
            systemRecorder.stop()
            print("stopped")
            exit(0)
        }
        source.resume()
        sources.append(source)
    }
    withExtendedLifetime(sources) { dispatchMain() }
}

// MARK: - main

func log(_ message: String) {
    FileHandle.standardError.write("ytcc-rec: \(message)\n".data(using: .utf8)!)
}

func usage() -> Never {
    log("usage: ytcc-rec detect | ytcc-rec record --mic FILE --system FILE [--echo-cancel]")
    exit(2)
}

let args = Array(CommandLine.arguments.dropFirst())
do {
    switch args.first {
    case "detect":
        try detect()
    case "record":
        var mic: String?, system: String?, echoCancel = false
        var rest = args.dropFirst().makeIterator()
        while let arg = rest.next() {
            switch arg {
            case "--mic": mic = rest.next()
            case "--system": system = rest.next()
            case "--echo-cancel": echoCancel = true
            default: usage()
            }
        }
        guard let mic, let system else { usage() }
        try record(mic: URL(fileURLWithPath: mic), system: URL(fileURLWithPath: system), echoCancel: echoCancel)
    default:
        usage()
    }
} catch {
    log("\(error)")
    exit(1)
}
