#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import heapq  
import pickle
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
punctuation = punctuation + '\n'
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier


# In[2]:


categories = ['Economy & Business', 'Diverse News', 'Politic', 'Sport', 'Technology']


# In[3]:


def nltk_summarizer(input_text, number_of_sentence):
    stopWords = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    word_frequencies = {}  
    for word in nltk.word_tokenize(input_text):  
        if word not in stopWords:
            if word not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(input_text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(number_of_sentence, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)  
    return summary


# In[4]:


en_data = pd.read_csv(r"C:/Users/USER/Desktop/Text-Summarizer-and-Categorical-main/Text-Summarizer-and-Categorical-main/dataset/bbc_news_dataset.csv")
en_data = en_data.replace("entertainment", "diverse news")
en_data = en_data.replace("business", "economy & business")

ar_data = pd.read_csv(r"C:/Users/USER/Desktop/Text-Summarizer-and-Categorical-main/Text-Summarizer-and-Categorical-main/dataset/arabic_dataset.csv")
ar_data = ar_data.replace("diverse", "diverse news")
ar_data = ar_data.replace("culture", "diverse news")
ar_data = ar_data.replace("politic", "politics")
ar_data = ar_data.replace("technology", "tech")
ar_data = ar_data.replace("economy", "economy & business")
ar_data = ar_data.replace("internationalNews", "politics")
ar_data = ar_data[~ar_data['type'].str.contains('localnews')]
ar_data = ar_data[~ar_data['type'].str.contains('society')]
en_data.sample(10)


# In[5]:


ar_data.sample(10)


# In[6]:


def delete_links(input_text):
    pettern  = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    out_text = re.sub(pettern, ' ', input_text)
    return out_text


# In[7]:


def delete_repeated_characters(input_text):
    pattern  = r'(.)\1{2,}'
    out_text = re.sub(pattern, r"\1\1", input_text)
    return out_text


# In[8]:


def replace_letters(input_text):
    replace = {"أ": "ا","ة": "ه","إ": "ا","آ": "ا","": ""}
    replace = dict((re.escape(k), v) for k, v in replace.items()) 
    pattern = re.compile("|".join(replace.keys()))
    out_text = pattern.sub(lambda m: replace[re.escape(m.group(0))], input_text)
    return out_text


# In[9]:


def clean_text(input_text):
    replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
    out_text = re.sub(replace, " ", input_text)
    words = nltk.word_tokenize(out_text)
    words = [word for word in words if word.isalpha()]
    out_text = ' '.join(words)
    return out_text


# In[10]:


def remove_vowelization(input_text):
    vowelization = re.compile(""" ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ""", re.VERBOSE)
    out_text = re.sub(vowelization, '', input_text)
    return out_text


# In[11]:


def delete_stopwords(input_text):
    stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    wnl = nltk.WordNetLemmatizer()
    lemmatizedTokens =[wnl.lemmatize(t) for t in tokens]
    out_text = [w for w in lemmatizedTokens if not w in stop_words]
    out_text = ' '.join(out_text)
    return out_text


# In[12]:


def stem_text(input_text):
    st = ISRIStemmer()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    out_text = [st.stem(w) for w in tokens]
    out_text = ' '.join(out_text)
    return out_text


# In[13]:


def text_prepare(input_text, ar_text):
    out_text = delete_links(input_text)
    out_text = delete_repeated_characters(out_text)
    out_text = clean_text(out_text)
    out_text = delete_stopwords(out_text)
    if ar_text:
        out_text = replace_letters(out_text)
        out_text = remove_vowelization(out_text)
        out_text = stem_text(out_text)
    else:
        out_text = out_text.lower()
    return out_text


# In[14]:


en_data['Processed Text'] = en_data['Text'].apply(text_prepare, args=(False,))
ar_data['Processed Text'] = ar_data['text'].apply(text_prepare, args=(True,))
en_data.sample(10)


# In[15]:


ar_data.sample(10)


# In[16]:


en_label_encoder = LabelEncoder()
en_data['Category Encoded'] = en_label_encoder.fit_transform(en_data['Category'])

ar_label_encoder = LabelEncoder()
ar_data['Category Encoded'] = ar_label_encoder.fit_transform(ar_data['type'])
ar_data['Category Encoded'] = ar_data['Category Encoded'].replace(1, 0)
ar_data['Category Encoded'] = ar_data['Category Encoded'].replace(0, 1)

en_data.sample(10)


# In[17]:


ar_data.sample(10)


# In[18]:


en_X_train, en_X_test, en_y_train, en_y_test = train_test_split(en_data['Processed Text'], en_data['Category Encoded'], test_size=0.2, random_state=0)
ar_X_train, ar_X_test, ar_y_train, ar_y_test = train_test_split(ar_data['Processed Text'], ar_data['Category Encoded'], test_size=0.2, random_state=0)


# In[19]:


def tfidf_features(X_train, X_test, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, ngram_range))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test


# In[20]:


en_features_train, en_features_test = tfidf_features(en_X_train, en_X_test, 2)


# In[21]:


ar_features_train, ar_features_test = tfidf_features(ar_X_train, ar_X_test, 2)


# In[22]:


def fit_model(model_name, ar_text=False):
    if model_name == 'ridge_model':
        model_name = RidgeClassifier()
    elif model_name == 'random_forest_model':
        model_name = RandomForestClassifier()
    elif model_name == 'logistic_regression_model':
        model_name = LogisticRegression()
    elif model_name == 'kneighbors_model':
        model_name = KNeighborsClassifier()
    elif model_name == 'decision_tree_model':
        model_name = DecisionTreeClassifier()
    elif model_name == 'gaussian_nb_model':
        model_name = GaussianNB()
    if ar_text:
        model_name.fit(ar_features_train.toarray(), ar_y_train)
        model_predictions = model_name.predict(ar_features_test.toarray())
        print("Accuracy on test: ", accuracy_score(ar_y_test, model_predictions))
    else:
        model_name.fit(en_features_train.toarray(), en_y_train)
        model_predictions = model_name.predict(en_features_test.toarray())
        print("Accuracy on test: ", accuracy_score(en_y_test, model_predictions))
    return model_name


# In[23]:


def summerize_category(input_text, statements, model_name, ar_text=False):
    summary_text = nltk_summarizer(input_text, statements)
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print("Text summary")
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print(summary_text)
    print("-------------------------------------------------------------------------------------------------------------------------------")
    input_text_arr = [text_prepare(input_text, ar_text)]
    if ar_text:
        features_train, features_test = tfidf_features(ar_X_train, input_text_arr, 2)
    else:
        features_train, features_test = tfidf_features(en_X_train, input_text_arr, 2)
    text_predection = model_name.predict(features_test.toarray())
    print("Text category:", categories[text_predection[0]])
    print("-------------------------------------------------------------------------------------------------------------------------------")


# In[24]:


en_ridge_model = fit_model('ridge_model')
pickle.dump(en_ridge_model, open('en_ridge_model.pkl','wb'))


# In[25]:


en_random_forest_model = fit_model('random_forest_model')
pickle.dump(en_random_forest_model, open('en_random_forest_model.pkl','wb'))


# In[26]:


en_logistic_regression_model = fit_model('logistic_regression_model')
pickle.dump(en_logistic_regression_model, open('en_logistic_regression_model.pkl','wb'))


# In[27]:


en_kneighbors_model = fit_model('kneighbors_model')
pickle.dump(en_kneighbors_model, open('en_kneighbors_model.pkl','wb'))


# In[28]:


en_decision_tree_model = fit_model('decision_tree_model')
pickle.dump(en_decision_tree_model, open('en_decision_tree_model.pkl','wb'))


# In[29]:


en_gaussian_nb_model = fit_model('gaussian_nb_model')
pickle.dump(en_gaussian_nb_model, open('en_gaussian_nb_model.pkl','wb'))


# In[30]:


test_1 = "The abstract will talk about the company Redbull, which is an international drink company that is widely known for its vitalizing energy drinks. One of Red bull’smany priorities is to enhance its digital presence , creating interesting content, using social media efficiently , and delving in e commerce. Chapter sixteen was involved in this paper , it involves (personal selling), and showing how important personal selling is in marketing communications. The use of this chapter is supported by the company’s marketing strategy. In Red bull’s marketing strategy personal selling is very vital in creating relationships with the customer , illustrating the brand’s image and values, and increasing sales. Personal selling is also justified by the AIDA model , social exchange theory and relationship marketing. These ideas offer insights into how customers make decisions, develop relationships, and exchange value with brands. These theories can help a company create marketing strategies efficiently. The AIDA model enables a thorough comprehension of how personal selling aids customers as they progress through the phases of decisionmaking. While the social exchange theory describes the reasons and dynamics underlying personal selling, relationship marketing offers insights into creating and maintaining long-term client connections. Applying these ideas will enable marketers to build plans that are in line with customer behaviorand preferences and, as a result, improve marketing execution. "


# In[31]:


summerize_category(test_1, 10, en_random_forest_model)


# In[32]:


ar_ridge_model = fit_model('ridge_model', True)
pickle.dump(ar_ridge_model, open('ar_ridge_model.pkl','wb'))


# In[33]:


ar_random_forest_model = fit_model('random_forest_model', True)
pickle.dump(ar_random_forest_model, open('ar_random_forest_model.pkl','wb'))


# In[34]:


ar_logistic_regression_model = fit_model('logistic_regression_model', True)
pickle.dump(ar_logistic_regression_model, open('ar_logistic_regression_model.pkl','wb'))


# In[35]:


ar_kneighbors_model = fit_model('kneighbors_model', True)
pickle.dump(ar_kneighbors_model, open('ar_kneighbors_model.pkl','wb'))


# In[36]:


ar_decision_tree_model = fit_model('decision_tree_model', True)
pickle.dump(ar_decision_tree_model, open('ar_decision_tree_model.pkl','wb'))


# In[37]:


ar_gaussian_nb_model = fit_model('gaussian_nb_model', True)
pickle.dump(ar_gaussian_nb_model, open('ar_gaussian_nb_model.pkl','wb'))


# In[47]:


ar_test_1 = "مع تبوأ بيب غوارديولا دفة الإدارة الفنية لبرشلونة تغيرت أمور كثيرة في الفريق سواء بانضباط اللاعبين أو بأدائهم داخل الملعب. خلال عامه الأول مع برشلونة حقق غوارديولا ما لم يحققه أي مدرب في العالم، ففي تاريخ 2 مايو عام 2009، خاض برشلونة مبارة الكلاسيكو مع غريمه التقليدي ريال مدريد في معقل الأخير، ملعب سانتياغو برنابيو، وحقق برشلونة حينها انتصارًا مدويًا، ففاز بنتيجة 6-2. ضمنت تلك النتيجة إلى حد كبير فوز برشلونة بلقب الليغا، وبعدها بأسبوعين التقى برشلونة مع أتلتيك بلباو في نهائي كأس إسبانيا، وحقق برشلونة اللقب الذي كان غائبا عن خزائنه مدة 13 عامًا، وفي أواخر ذات الشهر حقق برشلونة لقب دوري أبطال أوروبا على حساب نادي مانشستر يونايتد الإنجليزي، وليكون ذلك اللقب الثالث للنادي بتلك البطولة. في شهر أغسطس من ذلك العام ظفر برشلونة ببطولتي كأس السوبر الإسباني على حساب أتلتيك بيلباو للمرة الثامنة بتاريخه وبطولة كأس السوبر الأوروبي على حساب نادي شاختار دونستيك الأوكراني، وفي أواخر العام ذاته شارك النادي كممثل لقارة أوروبا في بطولة كأس العالم لأندية كرة القدم محققًا لقبها لأول مرة في تاريخه بعد انتصاره في المباراة النهائية على نادي إستوديانتيس دو لا بلاتا الأرجنتيني في نهائي مثير امتد لشوطين إضافيين، ليسدل الفريق الستار عن ذلك العام الاستثنائي بإنجاز غير مسبوق، بلغ 6 ألقاب بعام واحد، ويعرف باسم السداسية التاريخية. من أهم النجوم خلال ذلك العام: ليونيل ميسي، تشافي هيرنانديز، أندريس إنيستا رغم أن الغلة خلال عام 2010 لم تكن كسابقتها إلا أن الإبداع والأداء الراقي ظل مستمرًا. أبرز ما حدث في ذلك العام من انجازات كان الظفر ببطولة الليغا للمرة العشرين بتاريخ النادي، وكأس السوبر الإسباني للمرة التاسعة، وخروج برشلونة من دوري أبطال أوروبا أمام إنتر ميلان الإيطالي من الدور نصف النهائي، وخرج كذلك مبكرًا من كأس الملك أمام نادي إشبيلية. أما على صعيد الانتقالات، فقد غادر خلال صيف ذلك العام الفرنسي تيري هنري، وسبقه في الرحيل الكميروني صامويل ايتو في صفقة مبادلة مع نادي الانتر الإيطالي أتى بموجبها اللاعب السويدي إبراهيموفيتش في صفقة تعد الأكبر بتاريخ النادي. لم يمكث ابراهيموفيتش سوى عام واحد أعير بعده إلى لنادي ميلان الإيطالي، وضُم المهاجم الإسباني دافيد فيا من نادي فالنسيا خلال صيف عام 2010. إداريًا فقد جرت انتخابات رئاسة للنادي، وصل على أثرها ساندرو روسيل لرئاسة النادي خلفًا للمحامي خوان لابورتا. في افتتاحية موسم 2010-2011 حقق النادي لقب كأس السوبر الإسباني للمرة التاسعة في تاريخه، وشهد ذات الموسم صراعًا محتدمًا بين برشلونة وغريمه ريال مدريد، ولم يقتصر هذا الصراع على بطولة الليغا، التي حافظ عليها الفريق للمرة الثالثة على التوالي والحادية والعشرين في تاريخ النادي، إذ امتد الصراع ليشمل بطولتي كأس الملك ودوري أبطال أوروبا. وصل برشلونة المباراة النهائية ببطولة كأس إسبانيا ليلتقي الريال الذي استطاع خطف هدف الفوز بالوقت الإضافي الأول، وبعد تلك المباراة بأقل من أسبوع التقى برشلونة مجددًا مع ريال مدريد في الدور نصف النهائي لبطولة دوري أبطال أوروبا واستطاع برشلونة الإطاحة بريال مدريد والوصول للمباراة النهائية بتلك البطولة، والتي جمعت برشلونة مع نادي مانشستر يونايتد الإنجليزي، على ملعب ويمبلي في العاصمة البريطانية لندن، انتصر على إثرها فريق المدرب بيب غوارديولا بنتيجة 3-1 بعد أداء خيالي لنجوم الفريق في المباراة النهائية، محققين اللقب الرابع للكتلان بتلك البطولة، ومحققين أيضًا رقما قياسيا في عدد مرات الوصول لمبارة نهائية في إطار بطولات الأندية الأوروبية. لم تتوقف إنجازات النادي خلال عام 2011 عند ذلك، ففي مستهل الموسم الكروي 2011- 2012 وتحديدًا في منتصف شهر أغسطس، فاز النادي بكأس السوبر الإسباني عندما تفوق على غريمة التقليدي فريق ريال مدريد، ولم يمض أسبوع بعد تحقيق لقب تلك البطولة حتى حقق النادي كأس السوبر الأوروبي على حساب نادي بورتو البرتغالي، لتزداد غلة النادي من الألقاب ويصبح جوسيب غوارديولا أنجح مدرب في تاريخ النادي من حيث عدد الألقاب، وفي أواخر عام 2011 شارك النادي ببطولة كأس العالم للأندية في اليابان وتمكن من الظفر بلقب تلك البطولة بعد تفوقة في المباراة النهائية على فريق سانتوس البرازيلي"


# In[48]:


summerize_category(ar_test_1, 4, ar_ridge_model, True)


# In[ ]:




