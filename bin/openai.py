# def summarize_text(text):
#     response = openai_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Respond with a modification of the story based on the user's prompt, altering the context, characters, or storyline as necessary."
#             },
#             {
#                 "role": "user",
#                 "content": f"Summarize the following text:\n\n{text} and Identify the main characters"
#             }
#         ],
#         temperature=0.7,
#         max_tokens=100,
#         top_p=1
#     )
#
#     summary = response.choices[0].message.content
#     print('Summary : ' + summary)
#     return summary
#
#
# def extract_main_characters(text):
#     response = openai_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Respond to the queries of the story based on the user's prompt."
#             },
#             {
#                 "role": "user",
#                 "content": f"Identify the main characters in the following text:\n\n{text}"
#             }
#         ],
#         temperature=0.7,
#         max_tokens=100,
#         top_p=1
#     )
#     characters = response.choices[0].message.content
#     print('Characters : ' + characters)
#     return characters.split(', ')
#
#
# def chat_with_embeddings(text):
#     response = openai_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Respond with a modification of the story based on the user's prompt, altering the context, characters, or storyline as necessary."
#             },
#             {
#                 "role": "user",
#                 "content": f"{text}." + "Change the characters name Romeo and Juliet to ramanan, chandrika and change the context like this story happened in kerala and both are from fisherman family"
#             }
#         ],
#         temperature=0.7,
#         max_tokens=1000,
#         top_p=1
#     )
#
#     chat_response = response.choices[0].message.content
#     print("Response Chat : " + chat_response)
#     return chat_response
