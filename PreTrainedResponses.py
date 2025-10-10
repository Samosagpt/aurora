RESPONSES = {
    'who are you': "I am Aurora, your virtual assistant...",
    'are you fine': "I am doing fine and always at your service.",
    'help': "I can search Wikipedia, open websites, play music, and more!",
}


def get_pretrained_response(query):
    key = query.lower().strip()
    return RESPONSES.get(key)