import re 
import numpy as np
import pandas as pd

class Preprocess():
    def __init__(self, top_100_coins_path) -> None:
        self.coins=pd.read_csv(top_100_coins_path)
        self.coins['with$']=self.coins['symbol'].apply(lambda x:'$'+x+' ')
        self.coins['with#']=self.coins['symbol'].apply(lambda x:'#'+x+' ')
        self.coins['symbol']=self.coins['symbol'].apply(lambda x:' '+x.upper()+' ')

    def normalizeToken(self, token):
        """return normal tweet with mentions and http url
        Args:
            token ([string]): [a word(a token)]
        Returns:
            [string]: [normalized token]
        """
        if token.startswith("@"):
            return "@user"
        elif token.lower().startswith("http") or token.lower().startswith("www"):
            return "httpurl"
        else:
            return (token)

    def unicoin(self, tweet, dolar):
        tokens=tweet.split()
        for i,w in enumerate(tokens):
            if w[0]=='$' or w[0]=='#':
                w=w[1:]
            if w in dolar or w in list(self.coins['coin_name']) or w in list(self.coins['symbol']):
                tokens[i]='a_coin_name'
        return ' '.join(tokens)

    def unitext(self, tweets):
        dolars = tweets.apply(lambda x:set([ t.replace('$','') for t in x.split() if t.startswith('$')and len(t)>1 ]))

        # tweets = tweets.apply(lambda x: (" ".join([self.normalizeToken(token) for token in x.split()])))
        tweets = tweets.apply(lambda x: re.sub(r"\d+", " a_number ", str(x)))
        ut=[]
        for i in range(len(tweets)):
            ut.append(self.unicoin(tweets[i],dolars[i]))
        return np.array(ut)  

    def preprocess(self, texts):
        data=pd.DataFrame()
        data['texts']=texts
        data['texts'] = data['texts'].apply(lambda x: x.replace('&amp;','and').lower())
        data['unicode_text_concat']=self.unitext(data['texts'])
        return data['unicode_text_concat']