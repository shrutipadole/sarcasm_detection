import wget
import pandas as pd
import gzip
import json
import shutil
import re
import numpy as np
import os
import sys
from datetime import datetime
# from gensim.parsing.preprocessing import remove_stopwords

import string
LANGUAGE = string.ascii_lowercase + string.punctuation + ' '

import nltk
from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer
LEMMATIZER = WordNetLemmatizer()


# from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, RegexpStemmer 
# ps = PorterStemmer()
# # ps = SnowballStemmer(language='english')
# # ps = LancasterStemmer()
# # ps = RegexpStemmer('ing$|s$|e$|able$', min=4)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

SKIP_FILES = ["amazon_reviews_us_Wireless_v1_00.tsv", "amazon_reviews_us_Watches_v1_00.tsv",
"amazon_reviews_us_Video_Games_v1_00.tsv"]

with open('../data/source/common_abbreviations.json') as user_file:
  COMMON_ABBREV = json.loads(user_file.read())


def download_file(url):
    """
    input : url to zip files of amazon customer reviews
            Eg: "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Wireless_v1_00.tsv.gz"
    output : path to downloaded file
             Eg: '../data/dataset/sample_us.tsv'
    """
    # import pdb;pdb.set_trace()
    path = '../data/dataset'
    zip_path = path + '/' + url.split('/')[-1]
    unzip_path = zip_path.replace(".gz",'')

    if not os.path.exists(zip_path): #zip file not present 
        if not os.path.exists(unzip_path):
            print("Downloading ZIP file becasue File does not exist")
            filename = wget.download(url,out = path)
            print(filename)
            print("Dowload Complete\n\nInitiating Decompression")
            tsv_path = unzip(filename)
            print(tsv_path)
            print("Decompression Complete")
            # return filename
        else:
            print("Zip File Already Downloaded and Decompressed")
            tsv_path = unzip_path
    else: # zip file present already
        if os.path.exists(unzip_path):
            print("Zip File Already Downloaded and Decompressed")
            tsv_path = unzip_path
        else:
            print("Zip File Already Downloaded.\n\nInitiating Decompression")
            tsv_path = unzip(zip_path)
            print(tsv_path)
            print("Decompression Complete")

    return tsv_path

def delete(filename):
    """
    input : filepath to be deleted
            Eg: ""
    output : None
    """
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file does not exist:\t" + filename)
    return None

def unzip(infile):
    """
    input : just the filename to be unzipped. 
            Accpeted file type : .gz
            Eg: ".gz"
    output : name of unzipped file
             Eg:
    """
    # infile = '../data/dataset/' + filename.split('/')[-1]
    tofile = infile.replace('.gz', '')
    if infile.endswith('.gz'):
        with open(tofile, 'wb') as f_out, gzip.open(infile, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
        delete(infile)
    return tofile

def load_dataset(filename):
    """
    input : path of tsv file to be loaded. 
            Eg: ""
    output : dataframe with tsv dataset loaded
    """
    # import pdb;pdb.set_trace()
    ## based on pandas version , one of the two syntaxes below should run
    df = pd.read_csv(filename, sep='\t', index_col = False, on_bad_lines='skip')
    # df = pd.read_csv(filename, sep='\t', index_col = False, error_bad_lines='skip')
    print(df.shape)
    return df

def preprocess(sent):
    #converting all tokens to lowercase
    sent = str(sent).lower()
    #replace_acronyms
    sent = " ".join([COMMON_ABBREV[each] if each in COMMON_ABBREV else each for each in sent.split()])
    #replacing website links
    sent = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", sent)
    #removing stopwords
    #sent = remove_stopwords(sent)
    #stemming
    # sent = " ".join([ps.stem(word) for word in word_tokenize(sent)])
    #lemmatizing
    sent = " ".join([LEMMATIZER.lemmatize(word) for word in word_tokenize(sent)])
    #to filter out unicode or foreign characters
    sent = "".join([char for char in sent if char in LANGUAGE])
    #removing 's occurences
    sent = sent.replace("'s", "")
    #replacing words ending in "n't" with "word not"
    sent = sent.replace("wont", "will not").replace("won't", "will not").replace("cant", "can not")
    sent = sent.replace("can't", "can not")
    sent = re.sub(r"([a-zA-z])(n't)", r"\1 not", sent)
    sent = re.sub(r"(would|could|is|should)(nt)", r"\1 not", sent)
    #remove all HTML tags
    sent = re.sub(r"<.*?>", " ", sent)
    #replacing multiple punctuations with single occurrence
    sent = re.sub(r"\.+(\s?\.+)*", ".", sent)
    sent = re.sub(r"\!+", "!", sent)
    sent = re.sub(r"=+", "=", sent)
    sent = re.sub(r"_+", "_", sent)
    #introducing space after punctuation if not already there
    sent = re.sub(r"([,\.\"'\?\-:\(\)\\\/=\*\!_&#])(?!\s)", r"\1 ", sent)
    #introducing space before punctuation if not already there
    sent = re.sub(r"(?<!\s)([,\.\"'\?\-:\(\)\\\/=\!\*_])", r" \1", sent)
    #replacing words with more than 2 consecutive same character with two same characters
    sent = re.sub(r"([a-z])\1{1,}", r"\1\1", sent)
    #replacing multiple spaces with single space
    sent = re.sub(r"\s+", r" ", sent)
    
    return sent

def call_clean_df(df):
    dt_started = datetime.utcnow()
    print(df.shape)
    #applying cleaning procedures to review_headline and review_body

    # df['clean_review_headline'] = df.apply(lambda df: preprocess(df['review_headline']), axis=1)
    # df['clean_review_body'] = df.apply(lambda df: preprocess(df['review_body']), axis=1)

    #df['clean_review_headline'] = np.vectorize(preprocess)(df['review_headline'])
    #df['clean_review_body'] = np.vectorize(preprocess)(df['review_body'])
    
    df['clean_review_headline'] = df['review_headline'].map(preprocess)
    df['clean_review_body'] = df['review_body'].map(preprocess)

    dt_ended = datetime.utcnow()
    print("Time taken : ", str((dt_ended - dt_started).total_seconds()))
    return df
    


def main(sample):
    """
    input : url to zip files of amazon customer reviews
            Eg: ""
    output : filename
             Eg:
    """
    # download_file = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/sample_us.tsv'
    # download_file = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Wireless_v1_00.tsv.gz'
    # import pdb;pdb.set_trace()
    if sample:
        data = ['https://s3.amazonaws.com/amazon-reviews-pds/tsv/sample_us.tsv']
    else: 
        with open('../data/source/data_urls.txt', 'r') as f:
            data = f.readlines()
            data = [each.strip('\n') for each in data]
            data = [each for each in data if each.split("/")[-1].replace(".gz","") not in SKIP_FILES]
    for url in data:
        print("URL >>>>   " + url)
        tsv_path = download_file(url)
        print("Loading dataset from >>> ", tsv_path)
        df = load_dataset(tsv_path)
        #removing rows with null in the following two columns
        print("BEFORE REMOVING NA IN review_body >>>>>")
        print(df.shape)
        df = df[df['review_body'].notna()]
        print("AFTER REMOVING NA IN review_body >>>>>")
        print(df.shape)
        df = df[df['review_headline'].notna()]
        print("AFTER REMOVING NA IN review_headline >>>>>")
        print(df.shape)
        tofile = tsv_path.replace('.tsv','_trim.tsv')
        df.to_csv(tofile, sep = '\t', index=False)
        print('created file >>  \t',tofile)

        #deciding number of splits for processing huge dataframe
        num_splits = df.shape[0]//100000
        for ind, sub_df in enumerate(np.array_split(df, num_splits)):
            sub_df = call_clean_df(sub_df)
            replace_sent = 'clean_part_{}.tsv'.format(str(ind))
            tofile = tsv_path.replace('.tsv', replace_sent)
            sub_df.to_csv(tofile, sep = '\t', index=False)
            print('created file >>  \t',tofile)





        

    
if __name__ == "__main__":
    args = sys.argv
    # print(args)
    sample = True
    if  len(args) < 2:
        print("Running with sample data")
        main(sample)
    else:
        sample = False
        if args[1].lower() == 'run':
            print("Running with real data")
            main(sample)
        else:
            print('Command not supported. Check Documentation')
