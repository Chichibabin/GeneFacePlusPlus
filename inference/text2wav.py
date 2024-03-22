import azure.cognitiveservices.speech as speechsdk

def text_to_speech(text,voice_name ="zh-CN-XiaoxiaoNeural",  filename="input_audio.wav"):
    # 替换以下字符串
    subscription_key = "0e7b4c67d785478396023b7c5ac60b9b"
    service_region = "eastus"

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
    speech_config.speech_synthesis_voice_name = voice_name  # 指定中文声音

    audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to [{}] for text [{}]".format(filename, text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# 示例用法
text_to_speech("您好！这是一条数字人嘴部闭合测试语音。。。。。。")
