import paddlehub as hub

model = hub.Module(name='lstm_tacotron2', output_dir='./lstm_result', speaker_audio='./data/President_Chai.wav')  # 指定目标音色音频文件
texts = [
    '语音的表现形式在未来将变得越来越重要$',
    '今天的天气怎么样$',  ]
wavs = model.generate(texts, use_gpu=False)

for text, wav in zip(texts, wavs):
    print('='*30)
    print(f'Text: {text}')
    print(f'Wav: {wav}')