
##AI BASED CHATBOT    (MINOR PROJECT)

#imporitng all the necessary library
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer



#collecting all the methods
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('wordnet')



#opening the corpus dataset
with open('NLP_corpus.txt','r',encoding='utf8',errors='ignore') as f:
    raw_data=f.read()
raw_data=raw_data.lower()


#tokenization of raw data
tokenize_sent=nltk.sent_tokenize(raw_data)        #into sentence
tokenize_word=nltk.word_tokenize(raw_data)        #into word


#Data pre-processing
def lem_token(tokens):        #lemmatization
    lemmitizer = WordNetLemmatizer()
    return [lemmitizer.lemmatize(token)for token in tokens]

remove_punct=dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return lem_token(nltk.word_tokenize(text.lower().translate(remove_punct)))

input_greeting =['hello','hey','hi','hola','salam','namastey']
output_greeting = ['hello', 'hey', 'hi', 'hola', 'salam', 'namastey']
def greeting(query):
    # greeting = greeting.lower()
    for i in query.split():
        if i.lower() in input_greeting:
            return random.choice(output_greeting)


def response(user_response):
    robo1_response=''
    tokenize_sent.append(user_response)

    TfidVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf= TfidVec.fit_transform(tokenize_sent)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if (req_tfidf==0):
        robo1_response=robo1_response+"I am sorry"
        return robo1_response
    else:
        robo1_response= robo1_response+tokenize_sent[idx]
        return robo1_response

#Driver code
flag=True
print('I am chatbot:- ')
while(flag==True):
    user_response= input("your query:- ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks'):
            flag=False
            print("BOT- You're welcome.")
        else:
            if (greeting(user_response)!=None):
                print("BOT- "+greeting(user_response))
            else:
                print("BOT: ", end="")
                print(response(user_response))
                tokenize_sent.remove(user_response)
    else:
        flag=False
        print("BOT- bye.")

