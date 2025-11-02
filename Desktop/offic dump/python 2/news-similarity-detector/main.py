import pandas as pd

fake = pd.read_csv('fake.csv')
fake['target']='fake'

real = pd.read_csv('true.csv')
real['target']='real'

df = pd.concat([fake,real],ignore_index=True)[['title','text','target']]

df = df.dropna().reset_index(drop=True)



from sklearn.feature_extraction.text import TfidfVectorizer

df['combined']=df['title']+" "+df['text']

vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['combined'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

threshold = 0.8
duplicates = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if cosine_sim[i, j] >= threshold:
            duplicates.append((i, j, cosine_sim[i, j]))

print("Potential duplicate article pairs:", len(duplicates))
for i, j, score in duplicates[:5]:
    print(f"\nArticle {i} & {j} → {score:.2f}")
    print("Title 1:", df.loc[i, 'title'])
    print("Title 2:", df.loc[j, 'title'])




