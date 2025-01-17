from crewai import Task

class PodcastTasks:
    def task1_intro(self, agent, topic):
        return Task(
            description=f'Introduce yourself with a Doctor + random name. Do Host introduction of the show and interesting applications of the {topic}. Not more than 5 sentences.',
            expected_output=f'A brief introduction of the podcast show where complex topics are explained in simple terms by an expert and a compelling overview of interesting applications of the {topic}. Not more than 5 sentences.',
            agent=agent
        )

    def task2_overview(self, agent, topic, conversation_history=""):
        return Task(
            description=f'Introduce yourself with a Doctor + random name. Provide a high-level overview of the {topic} and a brief overview of what will be discussed. Not more than 5 sentences.',
            expected_output=f'A concise overview of the {topic} and an outline of the main points to be discussed. Not more than 5 sentences.',
            agent=agent,
            # context={"conversation_history": conversation_history} # tried to rectify this but doesn't worl
        )

    def task3_host(self, agent, topic, conversation_history=""):
        return Task(
            description=f"Ask a thought-provoking question or share interesting perspectives on the expert's explanation in brief words like Lex Fridman and Joe Rogan without answering listener's questions. ",
            expected_output="Concise natural sentences that reply to expert's latest response, possibly with a thought-provoking question.",
            agent=agent,
            # context=[conversation_history]
        )
    
    def task4_expert(self, agent, topic, conversation_history=""):
        return Task(
            description=f"""
            Continue the explanation of {topic}, building upon the overview previously provided.
            1. Address any question or comment from the host.
            2. Introduce and explain one new concept or aspect of {topic} in depth.
            3. Use clear, concise language suitable for a general audience.
            4. Provide real-world examples or analogies to illustrate complex ideas.
            5. Highlight the significance or implications of this aspect within the broader context of {topic}.
            6. Conclude your explanation at a natural breaking point to allow for host interaction.

            Remember to maintain an engaging tone and pace suitable for a podcast format.
            """,
            expected_output=f"""
            A detailed explanation of one aspect of {topic} that:
            - Responds to any previous host comments or questions
            - Introduces and thoroughly explains a new concept
            - Uses clear language and illustrative examples
            - Connects the explanation to the broader topic
            - Ends at a point that invites further discussion or questions
            """,
            agent=agent,
            # context=[conversation_history]  
        )
    
    def task5_userQuery(self, agent, topic, conversation_history=""):
        return Task(
            description='Answer user questions directed to the domain expert',
            expected_output=f'Clear and detailed answers to user questions about the {topic}',
            agent=agent,
            # context=[conversation_history]  
        )