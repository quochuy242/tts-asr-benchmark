from ViStreamASR import StreamingASR

# Initialize ASR
asr = StreamingASR()

# Process audio file
for result in asr.stream_from_file("./linh_ref_long.wav"):
    if result["partial"]:
        print(f"Partial: {result['text']}")
    if result["final"]:
        print(f"Final: {result['text']}")
