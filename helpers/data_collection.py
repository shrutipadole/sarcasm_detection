import wget
import pandas as pd
import gzip
import json
import re
import os
import sys
from gensim.parsing.preprocessing import remove_stopwords

import string
LANGUAGE = string.ascii_lowercase + string.punctuation + ' '

# import nltk
from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer
LEMMATIZER = WordNetLemmatizer()


# from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, RegexpStemmer 
# ps = PorterStemmer()
# # ps = SnowballStemmer(language='english')
# # ps = LancasterStemmer()
# # ps = RegexpStemmer('ing$|s$|e$|able$', min=4)

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

with open('../data/source/common_abbreviations.json') as user_file:
  COMMON_ABBREV = user_file.read()


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
    unzip_path = zip_path.rstrip('.gz')

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
    tofile = infile.rstrip('.gz')

    with open(infile, 'rb') as inf, open(tofile, 'w', encoding='utf8') as tof:
        decom_str = gzip.decompress(inf.read()).decode('utf-8')
        tof.write(decom_str)
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
    sent = remove_stopwords(sent)
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
    for url in data:
        print("URL >>>>   " + url)
        tsv_path = download_file(url)
        # if not sample:
        #     tsv_path = unzip(filename)
        # else:
        #     tsv_path = filename
        # print(filename + '\t<<<<<>>>>>>\t' + tsv_path)
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
