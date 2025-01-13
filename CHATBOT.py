import json
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import nltk

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

with open('UNIVERSITIES.json', 'r') as file:
    university_data = json.load(file)

with open('INTENTS.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def filter_universities(sat_score, location, uni_type, difficulty):
    filtered_data = university_data

    if sat_score:
        filtered_data = [
            uni for uni in filtered_data
            if uni.get("Min SAT", 0) <= sat_score <= uni.get("Max SAT", 1600)
        ]

    if location:
        filtered_data = [
            uni for uni in filtered_data
            if location.lower() in uni.get("Location", "").lower()
        ]

    if uni_type:
        filtered_data = [
            uni for uni in filtered_data
            if uni_type.lower() == uni.get("Type", "").lower()
        ]

    if difficulty:
        filtered_data = [
            uni for uni in filtered_data
            if difficulty.lower() == uni.get("Category", "").lower()
        ]

    return filtered_data

def format_results(universities):
    if not universities:
        return "No universities match your criteria."

    results = "Here are the universities that match your criteria:\n"
    for uni in universities:
        results += (f"- {uni['University Name']}\n  Type: {uni['Type']}\n  Location: {uni['Location']}\n"
                    f"  SAT Range: {uni['Min SAT']} - {uni['Max SAT']}\n  Category: {uni.get('Category', 'N/A')}\n\n")
    return results

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'responses' in i and i['responses']:
                return random.choice(i['responses'])
            else:
                return "I don't have a response for that yet."
    return "I'm not sure how to respond to that."

def search_university(user_input):
    for uni in university_data:
        if uni['University Name'].lower() in user_input.lower():
            return (f"Here's the information about {uni['University Name']}:\n"
                    f"  Type: {uni['Type']}\n"
                    f"  Location: {uni['Location']}\n"
                    f"  SAT Range: {uni['Min SAT']} - {uni['Max SAT']}\n"
                    f"  Category: {uni.get('Category', 'N/A')}\n")
    return None

def chatbot():
    print("Welcome to the University Chatbot! I am here to help you Enter your requirements!")
    sat_score = int(input("Enter your SAT score (or leave blank by pressing Enter): ") or 0)
    location = input("Enter the location (or leave blank by pressing Enter): ").strip()
    uni_type = input("Enter the type (Public/Private, or leave blank by pressing Enter): ").strip()
    difficulty = input("Enter the difficulty level (Easy/Medium/Hard, or leave blank by pressing Enter): ").strip()

    filtered_results = filter_universities(sat_score, location, uni_type, difficulty)
    print("\n" + format_results(filtered_results))

    while True:
        user_input = input("\nAsk me more questions or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            print("Goodbye! Have a great day!")
            break

        university_info = search_university(user_input)
        if university_info:
            print(university_info)
        else:
            intents_list = predict_class(user_input)
            response = get_response(intents_list, intents)
            print(f"\n{response}")

if __name__ == "__main__":
    chatbot()
