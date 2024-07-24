import json

from openai import OpenAI
from urllib.request import urlretrieve


class StoryIllustrator:
    client: OpenAI

    def __init__(self, client: OpenAI):
        self.client = client

    def transcribe_audio(self, input_path: str) -> str:
        with open(input_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

        return transcription.text

    def get_prompt(self, transcription: str) -> str:
        summary = transcription.split(".")[0]

        prompt = (
            f"Generate a children's story illustration for the following: {summary}"
        )

        if len(prompt) > 1000:
            prompt = transcription.split(",")[0]

        return prompt

    def generate_illustration(self, prompt: str) -> str:
        response = self.client.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024", n=1
        )

        return response.data[0].url


if __name__ == "__main__":
    illustrator = StoryIllustrator(client=OpenAI())

    story_data = {}

    story_data["transcription"] = illustrator.transcribe_audio(
        input_path="audio/input/fables_01_02_aesop.mp3"
    )

    story_data["summary"] = illustrator.get_prompt(
        transcription=story_data["transcription"]
    )

    story_data["image_url"] = illustrator.generate_illustration(
        prompt=story_data["summary"]
    )

    print(story_data)

    with open("audio/data/fables_01_02_aesop.json", "w") as stream:
        json.dump(story_data, stream)

    with open("audio/transcripts/fables_01_02_aesop.txt", "w") as stream:
        stream.write(story_data["transcription"])

    urlretrieve(
        url=story_data["image_url"], filename="audio/covers/fables_01_02_aesop.png"
    )
