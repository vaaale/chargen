from textblob import TextBlob


#wiki = TextBlob("Python is a high-level, general-purpose programming language.")
wiki = TextBlob("The car is green")

print(wiki.tags)
print(wiki.noun_phrases)

