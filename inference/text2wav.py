import azure.cognitiveservices.speech as speechsdk

def text_to_speech(text,voice_name ="zh-CN-YunxiNeural",  filename="input_audio_m.wav"):
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
text_to_speech("需求背景，1、目前AI面试官的形象供应仅有2个，无法满足企业主个性化展示品牌形象的诉求。需要提供更多样、更个性化的数字人定制能力。包含形象、声音、背景等元素。2、短期使用第三方服务进行支持，单个数字人定制成本1000+每年，基于数字人制作视频5元/分钟，成本较高。需尽快使用自研能力代替。功能架构如下图所示。需求详情为：1. 数字人定制，2D形象定制，1、训练数据源：支持通过真人视频进行数字人的训练。2、数字人格式：2D、坐姿、半身形式。3、诉求：在保证效果的情况下，能够尽可能降低对视频拍摄的要求。该要求更多为拍摄设备、场地等，训练后无需人工调整，训练时长尽可能短。")
