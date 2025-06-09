from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
voices = client.list_voices(language_code="vi-VN").voices

print("Available Vietnamese voices:")
for v in voices:
    gender = texttospeech.SsmlVoiceGender(v.ssml_gender).name
    print(f"  • {v.name}  ({gender})")
