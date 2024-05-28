#
# def save_chat_history(user_id, chat_id, query, response):
#     chat_key = f"user:{user_id}:chats"
#     if not redis_client.exists(chat_key):
#         chat_history = [{"query": query, "response": response}]
#         redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
#     else:
#         chat_data = redis_client.hget(chat_key, chat_id)
#         if chat_data:
#             chat_history = json.loads(chat_data)
#             if isinstance(chat_history, list):
#                 chat_history.append({"query": query, "response": response})
#                 redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
#             else:
#                 chat_history = [{"query": query, "response": response}]
#                 redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
#         else:
#             chat_history = [{"query": query, "response": response}]
#             redis_client.hset(chat_key, chat_id, json.dumps(chat_history))
#
#
# def get_chat_history_for_user(user_id):
#     chat_key = f"user:{user_id}:chats"
#     if redis_client.exists(chat_key):
#         chat_history = redis_client.hgetall(chat_key)
#         chat_history = {chat_id.decode("utf-8"): json.loads(chat_data) for chat_id, chat_data in chat_history.items()}
#         return chat_history
#     else:
#         return {}
#
#
# def get_specific_chat(user_id, chat_id):
#     chat_key = f"user:{user_id}:chats"
#     if redis_client.exists(chat_key):
#         chat_data = redis_client.hget(chat_key, chat_id)
#         if chat_data:
#             return json.loads(chat_data.decode("utf-8"))
#     return None
