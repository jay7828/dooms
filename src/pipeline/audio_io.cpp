#include "pipeline/audio_io.h"

#include <AudioToolbox/AudioToolbox.h>

#include <fstream>

namespace dooms {

namespace {

constexpr UInt32 kReadFrames = 1 << 16;   // frames decoded per ExtAudioFileRead call

struct FileCloser {
    ExtAudioFileRef f = nullptr;
    ~FileCloser() { if (f) ExtAudioFileDispose(f); }
};

std::string os_status(OSStatus s) { return "OSStatus " + std::to_string(static_cast<long>(s)); }

}  // namespace

Result<std::vector<float>> load_audio(const std::string & path, int32_t sample_rate) {
    if (sample_rate < 1000 || sample_rate > 192000) return invalid_argument("sample_rate must be in [1000, 192000]");
    if (!std::ifstream(path, std::ios::binary)) return not_found("cannot open '" + path + "'");
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8 *>(path.c_str()), static_cast<CFIndex>(path.size()), false);
    if (!url) return invalid_argument("bad path '" + path + "'");
    FileCloser file;
    OSStatus st = ExtAudioFileOpenURL(url, &file.f);
    CFRelease(url);
    if (st != noErr) return data_loss("'" + path + "' is not audio AudioToolbox can decode (" + os_status(st) + ")");
    // Client format: mono float32 at the requested rate; AudioToolbox converts channels, sample format and rate while decoding.
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = sample_rate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mChannelsPerFrame = 1;
    fmt.mBitsPerChannel = 32;
    fmt.mBytesPerFrame = 4;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = 4;
    st = ExtAudioFileSetProperty(file.f, kExtAudioFileProperty_ClientDataFormat, sizeof fmt, &fmt);
    if (st != noErr) return data_loss("'" + path + "': cannot convert to " + std::to_string(sample_rate) + " Hz mono float (" + os_status(st) + ")");
    std::vector<float> out, buf(kReadFrames);
    while (true) {
        AudioBufferList list;
        list.mNumberBuffers = 1;
        list.mBuffers[0].mNumberChannels = 1;
        list.mBuffers[0].mDataByteSize = kReadFrames * 4;
        list.mBuffers[0].mData = buf.data();
        UInt32 frames = kReadFrames;
        st = ExtAudioFileRead(file.f, &frames, &list);
        if (st != noErr) return data_loss("'" + path + "': decoding failed (" + os_status(st) + ")");
        if (frames == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + frames);
    }
    if (out.empty()) return data_loss("'" + path + "' contains no audio");
    return out;
}

}  // namespace dooms
