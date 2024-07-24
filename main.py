import json
import os

from openai import OpenAI
from urllib.request import urlretrieve


class StoryIllustrator:

    def __init__(self):
        self.client = OpenAI()

    def process_file(self, input_path: str) -> str:
        file_path, file_name = os.path.split(input_path)

        print("Beginning transcription for", file_name)
        self.transcribe_audio(input_path=input_path)

        print("Extracting image prompt for", file_name)
        self.get_prompt(transcription=self.transcription)

        print("Generating illustration for", file_name)
        self.generate_illustration(prompt=self.prompt)

        return self.__dict__()

    def transcribe_audio(self, input_path: str) -> str:
        """Take audio of a story and transcribe text using OpenAI whisper-1 model

        args:
        input_path (str) - path to the audio file

        output (str) - transcription text of the audio file
        """

        with open(input_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

        self.transcription = transcription.text
        return self.transcription

    def get_prompt(self, transcription: str) -> str:
        """Get a prompt based on transcription that fits within dall-e prompt limit

        args:
        transcription (str) - transcribed audio text

        output (str) - a prompt that can be used with dall-e
        """

        # No need to bother shortening the transcript if it's already small enough
        if len(transcription) < 1000:
            summary = transcription

        else:
            # Grab the first few sentences to gather enough context for a prompt
            transcription_parts = transcription.split(". ")
            summary = ". ".join(transcription_parts[:4])

        # Add more detail to the prompt for more of a storybook feel
        prompt = (
            f"Generate a children's story illustration for the following: {summary}"
        )

        # if len(prompt) > 4000:
        #     prompt = transcription.split(",")[0]

        self.prompt = prompt
        return self.prompt

    def generate_illustration(self, prompt: str) -> str:
        """Generate the illustration based on the input prompt

        args:
        prompt (str) - a prompt for dall-e

        output (str) - the url of the generated illustration
        """

        response = self.client.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024", n=1
        )

        self.image_url = response.data[0].url
        return self.image_url

    def __dict__(self):
        output = {}

        for attr in ("transcription", "prompt", "image_url"):
            if hasattr(self, attr):
                output[attr] = getattr(self, attr)

        return output


if __name__ == "__main__":
    illustrator = StoryIllustrator()

    story_data = illustrator.process_file(
        input_path="audio/input/fables_01_03_aesop.mp3"
    )

    # story_data = {}

    # story_data["transcription"] = illustrator.transcribe_audio(
    #     input_path="audio/input/fables_01_02_aesop.mp3"
    # )

    # story_data["summary"] = illustrator.get_prompt(
    #     transcription=story_data["transcription"]
    # )

    # story_data["image_url"] = illustrator.generate_illustration(
    #     prompt=story_data["summary"]
    # )

    print(story_data)

    with open("audio/output/data/fables_01_03_aesop.json", "w") as stream:
        json.dump(story_data, stream)

    with open("audio/output/transcripts/fables_01_03_aesop.txt", "w") as stream:
        stream.write(story_data["transcription"])

    urlretrieve(
        url=story_data["image_url"],
        filename="audio/output/covers/fables_01_03_aesop.png",
    )
