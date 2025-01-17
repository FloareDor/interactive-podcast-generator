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

load_dotenv()


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
    def __init__(self, topic, duration_minutes):
        self.topic = topic
        self.duration_minutes = duration_minutes
        self.conversation = []
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.tts = TextToSpeech()
        self.conversation_idx = 0

    async def run(self):
        agents = PodcastAgents()
        tasks = PodcastTasks()

        host = agents.Lex_Fridman()
        expert = agents.Domain_Expert()

        # Scrape Wikipedia data
        wiki_data = ""
        try:
            wiki_data = scrape_wikipedia(self.topic)
            self.conversation.append(f"Background data: {wiki_data}")
        except:
            wiki_data = ""
        print(wiki_data)

        # Introduction
        intro_task = tasks.task1_intro(host, self.topic)
        crew = Crew(agents=[host], tasks=[intro_task])
        intro_response = crew.kickoff()
        self.conversation.append(f"Host: {intro_response}")
        print(f"Host: {intro_response}")
        self.tts.save_audio_host(intro_response, f"{self.conversation_idx}.mp3")
        self.conversation_idx += 1
        # Overview
        overview_task = tasks.task2_overview(expert, self.topic)
        crew = Crew(agents=[expert], tasks=[overview_task])
        overview_response = crew.kickoff()
        self.conversation.append(f"Expert: {overview_response}")
        print(f"Expert: {overview_response}")
        self.tts.save_audio_expert(overview_response, f"{self.conversation_idx}.mp3")
        self.conversation_idx += 1
        
        # Main discussion
        while datetime.now() < self.end_time:
            # Host's turn
            host_task = tasks.task3_host(host, self.topic)
            host_task.description += f" Current conversation: {self.conversation}"
            crew = Crew(agents=[host], tasks=[host_task])
            host_response = crew.kickoff()
            self.conversation.append(f"Host: {host_response}")
            print(f"Host: {host_response}")
            self.tts.save_audio_host(intro_response, f"{self.conversation_idx}.mp3")
            self.conversation_idx += 1

            if await self.handle_listener_interaction(expert):
                continue

            # Expert's turn
            expert_task = tasks.task4_expert(expert, self.topic)
            expert_task.description += f" Current conversation: {self.conversation}"
            crew = Crew(agents=[expert], tasks=[expert_task])
            expert_response = crew.kickoff()
            self.conversation.append(f"Expert: {expert_response}")
            print(f"Expert: {expert_response}")
            self.tts.save_audio_expert(overview_response, f"{self.conversation_idx}.mp3")
            self.conversation_idx += 1

            if await self.handle_listener_interaction(expert):
                continue

            if datetime.now() >= self.end_time:
                print("\nPodcast duration reached. Ending the podcast.")
                break

        return self.conversation

    async def handle_listener_interaction(self, expert):
        listener_input = input("\nDo you want to ask a question or comment? (y/n): ")
        if listener_input.lower() == 'y':
            question = input("Your question/comment: ")
            self.conversation.append(f"Listener: {question}")
            
            answer_task = Task(
                description=f"Answer the listener's question/comment: {question}. Current conversation: {self.conversation}",
                agent=expert,
                expected_output="A clear and concise answer to the listener's question or comment, relating it to the ongoing discussion"
            )
            crew = Crew(agents=[expert], tasks=[answer_task])
            answer = crew.kickoff()
            self.conversation.append(f"Expert: {answer}")
            print(f"Expert: {answer}")
            self.tts.save_audio_expert(answer, f"{self.conversation_idx}.mp3")
            self.conversation_idx+=1
            return True
        return False

async def main():
    print("##")
    print("-------------------------------")
    topic = input("Enter Desired topic: ")
    duration = int(input("Enter Desired time duration in minutes: "))

    custom_crew = PodcastCrew(topic, duration)
    conversation = await custom_crew.run()

    print("\nFull Conversation:")
    for message in conversation:
        print(message)

if __name__ == "__main__":
    asyncio.run(main())