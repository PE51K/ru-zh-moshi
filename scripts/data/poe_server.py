from poe_api_wrapper import PoeServer

tokens = [
    {
        'p-b': "fFISEyv4ZTIVZQkT7Pl-ZQ%3D%3D",
        'p-lat': "7JKu%2Bb0R6o%2FIZcRbg4U3nPrnEVY5eGnXGnAWs%2F4TyQ%3D%3D",
    }
]
PoeServer(tokens=tokens)

# You can also specify address and port (default is 127.0.0.1:8000)
PoeServer(tokens=tokens, address="0.0.0.0", port="8080")