import pandas as pd
from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization")

class ChatBotMemory:
    def __init__(self, transcript_file='transcript.csv', summary_file='summary.csv', memory_interval=25):
        self.transcript_file = transcript_file
        self.summary_file = summary_file
        self.memory_interval = memory_interval
        self.transcript_data = []
        self.load_memories()

    def load_memories(self):
        # Load existing transcripts and summaries
        try:
            self.transcript_data = pd.read_csv(self.transcript_file).to_dict(orient='records')
        except FileNotFoundError:
            self.transcript_data = []
        try:
            self.summary_data = pd.read_csv(self.summary_file)
        except FileNotFoundError:
            self.summary_data = pd.DataFrame(columns=['summary', 'exchange_range'])

    def save_transcript(self, chat_id, user_message, bot_response):
        # Save conversation exchange
        self.transcript_data.append({
            'chat_id': chat_id,
            'user_message': user_message,
            'bot_response': bot_response
        })
        pd.DataFrame(self.transcript_data).to_csv(self.transcript_file, index=False)

        # Summarize and save to long-term memory every memory_interval exchanges
        if len(self.transcript_data) % self.memory_interval == 0:
            self.generate_summary()

    def generate_summary(self):
        # Generate a summary for the last memory_interval exchanges
        recent_exchanges = self.transcript_data[-self.memory_interval:]
        recent_text = " ".join([f"User: {ex['user_message']} Bot: {ex['bot_response']}" for ex in recent_exchanges])
        summary = summarizer(recent_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        # Append summary to long-term memory
        new_summary = pd.DataFrame({
            'summary': [summary],
            'exchange_range': [f"{len(self.transcript_data)-self.memory_interval+1}-{len(self.transcript_data)}"]
        })
        self.summary_data = pd.concat([self.summary_data, new_summary], ignore_index=True)
        self.summary_data.to_csv(self.summary_file, index=False)

    def get_memory_data(self):
        # Combine full transcripts and summaries for RAG
        transcript_df = pd.DataFrame(self.transcript_data)
        combined_data = pd.concat([transcript_df, self.summary_data], ignore_index=True)
        return combined_data