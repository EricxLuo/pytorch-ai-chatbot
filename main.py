import os
import json
import random

import nltk #tokenization and lemetization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Run this line once and then delete
#nltk.download('punkt_tab')
#nltk.download('wordnet')

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fclayer1 = nn.Linear(input_size,128)
        self.fclayer2 = nn.Linear(128,64)
        self.fclayer3 = nn.Linear(64,output_size)

        self.relu = nn.ReLU() #Break linearity
        self.dropout = nn.Dropout(0.5)

    
    def forward(self, x):
        x = self.relu(self.fclayer1(x))
        x = self.dropout(x)
    
        x = self.relu(self.fclayer2(x))
        x = self.dropout(x)

        x = self.fclayer3(x)

        return x


class ChatbotHelper:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path
        
        self.doc = []
        self.vocab = []
        self.intents = []
        self.intents_response = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

        self.waiting_stocks= False
    


    @staticmethod
    def tokenizeLemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words =  nltk.word_tokenize(text)
        processed = []

        for word in words:

            lemmatized = lemmatizer.lemmatize(word.lower())
            processed.append(lemmatized)

        return processed
    
 
    def bagOfWords(self,words):
        bag=[]
        for word in self.vocab:
            if word in words:
                bag.append(1)
            else:
                bag.append(0)
        return bag
            
    #get data from intents.json and prepare
    def parseIntents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)
           

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_response[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenizeLemmatize(pattern)
                    self.vocab.extend(pattern_words)
                    self.doc.append((pattern_words,intent['tag']))
                
                self.vocab = sorted(set(self.vocab))



    #take all the words that were processed frm parse
    def prepareData(self):
        bags=[]
        i=[]

        for d in self.doc:
            words = d[0]
            bag = self.bagOfWords(words)

            intent_index = self.intents.index(d[1])
            bags.append(bag)
            i.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(i)



    #batchsize - how many instances process in parallel at once
    #lr - learning rate how quickly the model moves into the steepest descent
    #epoch - how many times we see same data
    def trainModel(self,batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32) #x = bagofword representation of all sentences
        y_tensor = torch.tensor(self.y, dtype=torch.long) # y= correct classification

        dataset = TensorDataset(X_tensor,y_tensor)
        loader = DataLoader(dataset,batch_size = batch_size, shuffle = True)

        #x.shape[1] dimensions of bagofwords
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion =  nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            runloss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs,batch_y) #comparing outputs to batch_y(the optimal/correct result)
                loss.backward() 
                optimizer.step()

                runloss +=loss
            print(f"Epoch {epoch+1}: Loss: {runloss/len(loader):.4f}")
        
    def saveModel(self,model_path,dimensions_path):
        torch.save(self.model.state_dict(),model_path)
        with open(dimensions_path, 'w') as file:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, file)



    def loadModel(self,model_path,dimensions_path):
        with open (dimensions_path,'r') as file:
            dimensions = json.load(file)
        

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path,weights_only = True))




    def processMessage(self,input_message):

        if self.waiting_stocks:
            self.waiting_stocks = False
            return addStocks(input_message)




        words = self.tokenizeLemmatize(input_message)
        bag= self.bagOfWords(words)

        bag_tensor = torch.tensor([bag], dtype = torch.float32)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(bag_tensor)

        predicted_index = torch.argmax(prediction, dim=1).item()
        predicted_intent = self.intents[predicted_index]


        #if it exists within functions then call the function associated with the input
        #otherwise, make a random choice from the list of responses in json.
        if self.function_mappings:
            if predicted_intent in self.function_mappings:

                if predicted_intent =="add_stock":
                    self.waiting_stocks = True
                    return "Tell me which stock to add."

                else:
                    return self.function_mappings[predicted_intent]()

        if self.intents_response[predicted_intent]:
            return random.choice(self.intents_response[predicted_intent])
        else:
            return None
        





portfolio=[]
def getStocks():
    if portfolio:
        return("Your portfolio:" + ", ".join(portfolio))
    else:
        return("Your portfolio is empty. Please ask me to add some stocks") 
def addStocks(message):
    stock = message.upper()

    if stock not in portfolio:
        portfolio.append(stock)
        return(f"{stock} added.")
    else:
        return(f"{stock} is already in your portfolio")


if __name__ == '__main__':
    bot = ChatbotHelper('intents.json', function_mappings = {'show_stocks': getStocks, 'add_stock':addStocks})
    bot.parseIntents()
    bot.prepareData()
    bot.trainModel(batch_size=8, lr=0.001,epochs=100)
    bot.saveModel('chatbot_model.pth', 'dimensions.json')


    while True:
        message = input("Enter message('q' to quit): ")
        if message == "q":
            break

        print(bot.processMessage(message))