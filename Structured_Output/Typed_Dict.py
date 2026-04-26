from typing import TypedDict

class User(TypedDict):
    user_name:str
    email:str
    age:int

user:User={
    'user_name':'Nikhil',
    'email':'chavannikhil762@gmail.com',
    'age':"21"
}

print(user)