# !pip install transformers soundfile torch datasets 
import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        
    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        self.engine.setProperty('volume', volume)

    def set_voice(self, voice_index):
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_index].id)

    def save_audio(self, text, output_file_path):
        self.engine.save_to_file(text, output_file_path)
        self.engine.runAndWait()

    def save_audio_expert(self, text, output_file_path):
        self.set_rate(125)
        self.set_volume(1.0)
        self.set_voice(1)  # female
        self.save_audio(text, output_file_path)

    def save_audio_host(self, text, output_file_path):
        self.set_rate(125)
        self.set_volume(1.0)
        self.set_voice(0)  # male
        text = text.replace('*', '')
        text = text.replace("Host:", "")
        text = text.replace("Expert:", "")
        self.save_audio(text, output_file_path)

# Example usage:
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.save_audio_expert('''Logistic regression is a statistical method used for predicting the probability of a binary outcome (e.g., yes/no, true/false) based on one or more predictor variables. It is a powerful tool for classification tasks and is widely used in various fields, including healthcare, finance, and marketing.

**Key points to be discussed:**

* **The core concepts of logistic regression:** How it works, the underlying mathematical model, and the interpretation of the results.
* **Types of logistic regression:** Binary logistic regression (predicting a binary outcome) and multinomial logistic regression (predicting an outcome with multiple categories).
* **Applications of logistic regression:** Real-world examples of how logistic regression is used in various industries.
* **Advantages and limitations of logistic regression:** Understanding its strengths and weaknesses compared to other classification methods.
* **Building and evaluating a logistic regression model:** Practical steps involved in building, training, and evaluating a logistic regression model.''', "output_person1.mp3")
    tts.save_audio_host("Hello World from person 2!", "output_person2.mp3")