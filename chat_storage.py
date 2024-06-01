import redis
import json


class RedisChatManager:
    def __init__(self, host: str, port: int, db: int, password: str):
        self.client = redis.Redis(host=host, port=port, db=db, password=password)

    def save_chat(self, user_id, chat_id, query, response):
        """
        Save a chat message to the chat history for a specific user and chat session.

        Parameters:
        user_id (str): The ID of the user.
        chat_id (str): The ID of the chat session.
        query (str): The user's query.
        response (str): The assistant's response.
        """
        chat_key = f"redis:{user_id}:chats"

        # Check if the chat history for the user exists
        if not self.client.exists(chat_key):
            # If not, create a new chat history list
            chat_history = [{"query": query, "response": response}]
            self.client.hset(chat_key, chat_id, json.dumps(chat_history))
        else:
            # If it exists, retrieve the existing chat data for the chat_id
            chat_data = self.client.hget(chat_key, chat_id)
            if chat_data:
                # If chat data exists, append the new query and response to the list
                chat_history = json.loads(chat_data)
                if isinstance(chat_history, list):
                    chat_history.append({"query": query, "response": response})
                    self.client.hset(chat_key, chat_id, json.dumps(chat_history))
                else:
                    # If chat history is not a list, create a new list with the query and response
                    chat_history = [{"query": query, "response": response}]
                    self.client.hset(chat_key, chat_id, json.dumps(chat_history))
            else:
                # If no chat data exists for the chat_id, create a new entry
                chat_history = [{"query": query, "response": response}]
                self.client.hset(chat_key, chat_id, json.dumps(chat_history))

    def user_chats(self, user_id):
        """
        Retrieve all chat histories for a specific user.

        Parameters:
        user_id (str): The ID of the user.

        Returns:
        dict: A dictionary containing all chat histories for the user, with chat IDs as keys.
        """
        chat_key = f"redis:{user_id}:chats"

        # Check if the chat history for the user exists
        if self.client.exists(chat_key):
            # Retrieve all chat histories for the user
            chat_history = self.client.hgetall(chat_key)
            # Decode and parse the chat data
            chat_history = {chat_id.decode("utf-8"): json.loads(chat_data) for chat_id, chat_data in
                            chat_history.items()}
            print(chat_history)
            return chat_history
        else:
            return {}

    def get_chat(self, user_id, chat_id):
        """
        Retrieve a specific chat session for a user by chat ID.

        Parameters:
        user_id (str): The ID of the user.
        chat_id (str): The ID of the chat session.

        Returns:
        dict or None: The chat history for the specified chat session, or None if not found.
        """
        chat_key = f"redis:{user_id}:chats"

        # Check if the chat history for the user exists
        if self.client.exists(chat_key):
            # Retrieve the specific chat data for the chat_id
            chat_data = self.client.hget(chat_key, chat_id)
            if chat_data:
                # Decode and parse the chat data
                return json.loads(chat_data)
        return {}

    def clear_database(self):
        self.client.flushdb()

    def user_chat_keys(self):
        pass
