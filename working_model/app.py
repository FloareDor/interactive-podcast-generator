import asyncio
import os
from crewai import Crew, Agent, Task
from textwrap import dedent
from agents import PodcastAgents
from tasks import PodcastTasks
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tts import TextToSpeech
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import io
from pydub import AudioSegment
import json

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PodcastRequest(BaseModel):
    topic: str
    duration: int

class QuestionRequest(BaseModel):
    question: str

# Global variable to store the PodcastCrew instance
global_podcast_crew = None

def scrape_wikipedia(topic):
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find(id="mw-content-text")
        paragraphs = content.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])
        return text
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"

class PodcastCrew:
    def __init__(self, topic, duration_minutes, websocket: WebSocket):
        self.topic = topic
        self.duration_minutes = duration_minutes
        self.conversation = []
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.tts = TextToSpeech()
        self.conversation_idx = -1
        self.websocket = websocket
        self.audio_folder = "podcast_audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.agents = PodcastAgents()
        self.tasks = PodcastTasks()
        self.host = self.agents.Lex_Fridman()
        self.expert = self.agents.Domain_Expert()
        self.ready_for_next = asyncio.Event()

    async def run(self):
        # Scrape Wikipedia data
        wiki_data = scrape_wikipedia(self.topic)
        self.conversation.append(f"Background data: {wiki_data}\n")

        # Introduction
        intro_task = self.tasks.task1_intro(self.host, self.topic)
        crew = Crew(agents=[self.host], tasks=[intro_task])
        intro_response = crew.kickoff()
        await self.save_and_send_audio(intro_response, "host", f"{self.conversation_idx}.mp3")
        self.conversation.append(f"\nHost: {intro_response}")
        self.conversation_idx += 1

        # Overview
        overview_task = self.tasks.task2_overview(self.expert, self.topic)
        crew = Crew(agents=[self.expert], tasks=[overview_task])
        overview_response = crew.kickoff()
        await self.save_and_send_audio(overview_response, "expert", f"{self.conversation_idx}.mp3")
        self.conversation.append(f"\nExpert: {overview_response}")
        self.conversation_idx += 1

        # Send initial conversation
        await self.websocket.send_json({"status": "initial", "conversation": self.conversation})

        # Main discussion
        while datetime.now() < self.end_time:
            # Wait for the frontend to be ready
            await self.ready_for_next.wait()
            self.ready_for_next.clear()

            # Host's turn
            host_task = self.tasks.task3_host(self.host, self.topic)
            host_task.description += f" Current conversation: {self.conversation}"
            crew = Crew(agents=[self.host], tasks=[host_task])
            host_response = crew.kickoff()
            await self.save_and_send_audio(host_response, "host", f"{self.conversation_idx}.mp3")
            self.conversation.append(f"\nHost: {host_response}")
            self.conversation_idx += 1

            # Wait for the frontend to be ready
            await self.ready_for_next.wait()
            self.ready_for_next.clear()

            # Expert's turn
            expert_task = self.tasks.task4_expert(self.expert, self.topic)
            expert_task.description += f" Current conversation: {self.conversation}"
            crew = Crew(agents=[self.expert], tasks=[expert_task])
            expert_response = crew.kickoff()
            await self.save_and_send_audio(expert_response, "expert", f"{self.conversation_idx}.mp3")
            self.conversation.append(f"\nExpert: {expert_response}")
            self.conversation_idx += 1

            if datetime.now() >= self.end_time:
                print("\nPodcast duration reached. Ending the podcast.")
                break

        await self.websocket.send_json({"status": "completed", "conversation": self.conversation})
        return self.conversation

    async def save_and_send_audio(self, text, speaker, filename):
        filepath = os.path.join(self.audio_folder, filename)
        if speaker == "host":
            self.tts.save_audio_host(text, filepath)
        else:
            self.tts.save_audio_expert(text, filepath)
        
        with open(filepath, "rb") as audio_file:
            await self.websocket.send_bytes(audio_file.read())

    async def handle_listener_question(self, question):
        try:
            self.conversation = self.conversation[:-2]
        except:
            self.conversation = self.conversation[:-1]
        print("question: ", question)
        answer_task = Task(
            description=f"Don't answer any other listener's question except this one latest listener's question: {question}. Current conversation: {self.conversation[:self.conversation_idx]}",
            agent=self.expert,
            expected_output="A clear and concise answer to the listener's question, relating it to the ongoing discussion"
        )
        crew = Crew(agents=[self.expert], tasks=[answer_task])
        answer = crew.kickoff()
        await self.save_and_send_audio(answer, "expert", f"{self.conversation_idx}.mp3")
        self.conversation.append(f"\nListener: {question}")
        self.conversation.append(f"Expert: {answer}")
        self.conversation_idx += 1

        # Generate the next part of the conversation immediately
        host_task = self.tasks.task3_host(self.host, self.topic)
        host_task.description += f" Current conversation: {self.conversation}"
        crew = Crew(agents=[self.host], tasks=[host_task])
        host_response = crew.kickoff()
        await self.save_and_send_audio(host_response, "host", f"{self.conversation_idx}.mp3")
        self.conversation.append(f"\nHost: {host_response}")
        self.conversation_idx += 1

        return answer, host_response

@app.websocket("/generate-podcast")
async def generate_podcast(websocket: WebSocket):
    global global_podcast_crew
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        topic = data["topic"]
        duration = data["duration"]

        global_podcast_crew = PodcastCrew(topic, duration, websocket)
        
        # Start the podcast generation in a separate task
        podcast_task = asyncio.create_task(global_podcast_crew.run())

        while True:
            message = await websocket.receive_json()
            if message["type"] == "ready_for_next":
                global_podcast_crew.ready_for_next.set()

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        global_podcast_crew = None

# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     global global_podcast_crew
#     if global_podcast_crew is None:
#         return {"error": "No active podcast session"}
    
#     answer, host_response = await global_podcast_crew.handle_listener_question(request.question)
#     return {
#         "answer": answer,
#         "next_host": host_response,
#     }

@app.post("/ask")
async def ask_question(file: UploadFile = File(...)):
    global global_podcast_crew
    if global_podcast_crew is None:
        raise HTTPException(status_code=400, detail="No active podcast session")
    
    try:
        # Transcribe the audio
        response = await transcribe_audio(file)
        print("response:", response, type(response))
        json_str = response.body.decode('utf-8')
        # Parse the JSON string to a Python dictionary
        data = json.loads(json_str)
        question = data['transcription']
        print("Transcribed question:", question)
        
        if not isinstance(question, str):
            raise HTTPException(status_code=400, detail="Failed to transcribe the audio")
        
        # Handle the question
        answer, host_response = await global_podcast_crew.handle_listener_question(question)
        
        return {
            "question": question,
            "answer": answer,
            "next_host": host_response,
        }
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as they are already formatted correctly
        raise http_exc
    except Exception as e:
        # Log the error for debugging
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your question: {str(e)}")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    r = sr.Recognizer()
    audio_buffer = io.BytesIO()

    try:
        audio_buffer.write(await file.read())
        audio_buffer.seek(0)
        audio_segment = AudioSegment.from_file(audio_buffer)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        pcm_wav = io.BytesIO()
        audio_segment.export(pcm_wav, format="wav")
        pcm_wav.seek(0)
        with sr.AudioFile(pcm_wav) as source:
            audio_data = r.record(source)

        text = r.recognize_google(audio_data)
        print(text)
        return JSONResponse(content={"transcription": text})
    
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)