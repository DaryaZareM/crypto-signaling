from emoji import demojize
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import yaml

from BERTweetclass import BERTweetclass
from CustomDataset import CustomDataset
from Preprocess import Preprocess


class CryptoPositivity():
    def __init__(self,config_path) -> None:
        self.config = self.load_configs_path(config_path)
        self.coins = pd.read_csv(self.config['top_10_coins'])
        self.coins['label'] = self.coins['label'].apply(lambda x:x.lower())
        self.tokenizer_ = TweetTokenizer()
        self.MAX_LEN = 100
        self.berTweetModel_path='vinai/bertweet-base' #bertTweet
        self.tokenizer = AutoTokenizer.from_pretrained(self.berTweetModel_path)
        self.model_reddit=self.load_model(self.config['model_reddit_path'] )
        self.model_twitter=self.load_model(self.config['model_twitter_path'] )
        self.model_instagram=self.load_model(self.config['model_instagram_path'] )
        self.VALID_BATCH_SIZE = 4
        self.preprocess_obj=Preprocess(self.config['top_100_coins'])
        

    def load_configs_path(config_path):
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return (config)

    def load_model(self,model_path):
        model = BERTweetclass(self.berTweetModel_path)
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        model.eval()
        return model

    def relevant_coins(self, text):
        res=set()
        tokens = self.tokenizer_.tokenize((text).lower())
        for i,w in enumerate(tokens):
            if w[0]=='$' or w[0]=='#':
                w=w[1:]
            for i in range(len(self.coins)):
                if w ==(self.coins.iloc[i]['full_id']) or w ==(self.coins.iloc[i]['label']):
                    res.add(self.coins.iloc[i]['label'])
        return list(res)

    def validation(self,model,validation_l):
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(validation_l, 0):
                ids = data['ids']
                mask = data['mask']
                token_type_ids = data['token_type_ids']
                outputs = model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs

    def normalizeToken_(self, token):
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token) #TODO
        else:
            return token


    def normalizeTweet(self,tweet):
        tokens = self.tokenizer_.tokenize((tweet).lower())
        normTweet = " ".join([self.normalizeToken_(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return (" ".join(normTweet.split()))

    def get_signal(self,model,texts):
        dataset = pd.DataFrame()        
        dataset['tweet']=texts
        dataset['tweet_normalize']=dataset['tweet'].apply(self.normalizeTweet)
        test_set = CustomDataset(dataset, self.tokenizer, self.MAX_LEN)
        test_params = {'batch_size': self.VALID_BATCH_SIZE,
                       'num_workers': 0}

        test_loader = DataLoader(test_set, **test_params)
        
        signals = self.validation(model, test_loader)
        return signals

    def preprocess(self, texts):
        pt = self.preprocess_obj.preprocess(texts)
        return pt

    def twitter_positivity(self,texts):
        is_relevant=[]
        relevant_coins=[]
        for i in range(len(texts)):
            coin_vector=self.relevant_coins(texts[i])
            relevant_coins.append(coin_vector) #TODO
            is_relevant.append(1 if len(coin_vector) and not 'airdrop' in texts[i].lower() and not 'giveaway' in texts[i].lower()  else 0)

        preprocess_text=self.preprocess(texts)
        signals = self.get_signal(self.model_twitter, preprocess_text)
    
        return self.create_output_set(signals,is_relevant,relevant_coins)
    
    def instagram_positivity(self,texts):
        is_relevant=[]
        relevant_coins=[]
        for i in range(len(texts)):
            coin_vector=self.relevant_coins(texts[i])
            relevant_coins.append(coin_vector) #TODO
            is_relevant.append(1 if len(coin_vector) and not 'airdrop' in texts[i].lower() and not 'giveaway' in texts[i].lower()  else 0)

        # preprocess_text=self.preprocess(texts)
        preprocess_text=(texts)
        signals = self.get_signal(self.model_instagram, preprocess_text)
    
        return self.create_output_set(signals,is_relevant,relevant_coins)
    
    def reddit_positivity(self,texts):
        is_relevant=[]
        relevant_coins=[]
        for i in range(len(texts)):
            coin_vector=self.relevant_coins(texts[i])
            relevant_coins.append(coin_vector) #TODO
            is_relevant.append(1 if len(coin_vector) else 0)

        preprocess_text=self.preprocess(texts)
        signals = self.get_signal(self.model_reddit, preprocess_text)
    
        return self.create_output_set(signals,is_relevant,relevant_coins)

    
    def create_output_set(self, signal,is_relevant,relevant_coins):
        out_df=pd.DataFrame()
        out_df['Relevant'] = is_relevant
        out_df['Positivity_'] = signal
        out_df['coins'] = relevant_coins
        out_df['Positivity'] = out_df['Positivity_'].apply(lambda x:np.argmax(x))
        return out_df[['Relevant','Positivity','coins']]
