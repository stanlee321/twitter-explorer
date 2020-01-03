# Python scraper for twitter
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

import itertools
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import argparse


# Customize matplotlib
# matplotlib.rcParams.update(
#     {
#         'text.usetex': False,
#         'font.family': 'stixgeneral',
#         'mathtext.fontset': 'stix',
#     }
# )

# PIPELINE


class ProcessTwitter:
    def __init__(self, json_n, json_m, target_name="", data_path=""):
        
        self.data_path = data_path
        
        self.target_name = target_name.replace(" ", "")

        
        self.df_n = self.create_timed_df(pd.read_json(json_n, encoding='utf-8'))
        self.df_m = self.create_timed_df(pd.read_json(json_m, encoding='utf-8'))
        
        self.df_n.to_csv(f"{data_path}/{self.target_name}_n.csv", sep=",")
        self.df_m.to_csv(f"{data_path}/{self.target_name}_m.csv", sep=",")

        # usernames from N and M
        self.uniques, self.user_names = self.get_uniques_and_user_names(self.df_n)
        self.uniques_m, self.user_names_m = self.get_uniques_and_user_names(self.df_m)
        
        # Users how mention to TargetUser
        self.users_mention =  self.df_n[self.df_n["username"] != target_name]["username"].str.replace(" ", "").tolist()
        self.users_mention_m = self.df_m["username"].str.replace(" ", "").tolist()

        self.sorted_counts = self.create_sorted_counts(self.user_names)
        self.sorted_counts_m = self.create_sorted_counts(self.user_names_m )
        
        self.target_dfs = self.create_mentions_df(self.df_n, self.sorted_counts, self.user_names)
        
        self.split_date = self.create_split_datetime()
        
        # Target DF
        try: 
            self.df_target = self.create_df_per_day(self.target_dfs[self.target_name])
            ## Add filter column with splite_date
            self.df_target['status'] = np.where(self.df_target.index < self.split_date, 0, 1)
        
        except Exception as e:
            print(e)
            self.df_target = None

        self.stopw = ["la", 
                 "es", 
                 "me",
                 "que",
                 "hay",
                 "mi",
                 "https",
                 "com",
                 "html",
                 "twitter",
                 "pic",
                 "http",
                 "fb",
                 "www",
                 "watch",
                 "status",
                 "si","del",
                 "con","son",
                 "te","por", "le","lo","va","ni","él", "el", "de", "esa", "en", "se", "al", "su"]
        
        
    def get_uniques_and_user_names(self, df):
        
        user_names = df["username"].apply(self.deEmojify)
        #user_names = df_raw_gabo["username"]
        user_names = user_names.str.replace(" ", "")

        uniques = user_names.unique()
        return uniques, user_names

    def deEmojify(self, inputString):
        return inputString.encode("utf-8", 'replace').decode("utf-8")
    
    def get_counts(self, user_names):
        
        counts = {}
        for u in user_names:
            counts[u] = []

        for u in user_names:
            counts[u].append(u)
        for k,v in counts.items():
            counts[k] = len(v)
            
        return counts

    def get_sorted_tuits(self, counts):
        sorted_twitts =  sorted(counts.items(), key= lambda kv:(kv[1], kv[0]))[::-1][:25]
        
        for i, (k,v) in enumerate( sorted_twitts[1:]):
            print(str(i) + "-", "Nombre: ", k, "   |||",  "Retuits: ", v)
        return sorted_twitts
    
    def create_df_counts(self, sorted_twitts):
        
        df_x = pd.DataFrame(sorted_twitts, columns=["label", "count"])
        df_x["index"] = df_x.index

        df_x.index = list(df_x["label"])

        df_x = df_x.sort_values(['count'], ascending=False)

        return df_x
    
    def plot_word_count(self, df, titley='Number of Retuits', titlex="Nombres", title="Frecuencia de mensajes"):
        plt.figure(figsize=(10,6))
        ax = sns.barplot(x="index", y="count", data=df, order=df['index'])
        ax.set_xlabel(titlex)
        ax.set_ylabel(titley)
        ax.set_xticklabels(df['label'], rotation='vertical', fontsize=10)  

        plt.savefig(self.data_path + "/" + title + ".png", format="png", bbox_inches = 'tight')


    def create_word_cloud(self, uniques, file_name="wordCloud.png"):
        # Create and generate a word cloud image:

        text = ' '.join(uniques)
        wordcloud = WordCloud(max_font_size=50, 
                                    max_words=100, 
                                    background_color="white", 
                                    width=600, height=460,
                                    stopwords=self.stopw).generate(text)

        # Display the generated image:
        plt.figure(figsize = (21,13))

        plt.imshow(wordcloud, interpolation='bilinear')

        plt.axis("off")
        plt.savefig(self.data_path + "/" + file_name, format="png")
        plt.show()
        
    def create_mentions_df(self, df, sorted_twitts, user_names):
        target_df = {}

        for st in sorted_twitts:
            aux_df = df[user_names == st[0]]
            target_df[st[0]] = aux_df
            
            
        return target_df
    
    def get_hashtag_list(self, df):
        hashtags_raw = df["hashtags"].tolist()

        clean_hashtags = []
        
        for i, h in enumerate(hashtags_raw):
            if len(h) == 0:
                pass
            else:
                clean_hashtags.append(h)

        hashtags = (list(itertools.chain.from_iterable(clean_hashtags)))

        return hashtags
    
    def create_timed_df(self, df):
        # Complete the call to convert the date column
        df["time"] =  pd.to_datetime(df["timestamp"],format='%Y-%m-%d %H:%M:%S')
        
        df.set_index('time', inplace=True)
        
        return df
    

    def create_df_per_day(self, df):
        
        df_d = df.groupby(df.index.floor('h')).size().reset_index(name='count')
        df_d.set_index('time', inplace=True)
        return df_d
    
    def create_split_datetime(self, year=2019, month=10, day=20, hour=0, minute=0 ):
        
        return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
            
    def get_timediff(self,df, status=0):
        
        time_diff = df[df["status"] != status].index.to_series().diff()
        return time_diff
    
    
    def plot_df_thrend(self, df, status=0):
        df_split_month = self._split_df(df, status)
        
        self.plot_thrend(df_split_month)
        
    def plot_df_thrend_per_hour(self, df, status=0, plot_name="df_thrend_per_hour.jpg"):
        
        df_split_month = self._split_df(df, status)
        
        per_hour = df_split_month.groupby(df_split_month.index.hour).mean()

        self._plot_df_xy( per_hour.index, per_hour["count"], plot_name)
        


    def _plot_df_xy(self, x, y, plot_name ):
          ## Plot part
        fig = plt.figure(figsize=(13,5))
        plt.title('Tuits Count')
        plt.xlabel('Hora')
        plt.ylabel('Tuits')

        plt.plot( x, y,  marker='o', label='Tuits promedio')
        plt.grid(color='r', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(self.data_path + "/" + plot_name )
        plt.show()


    def _split_df(self, df, split):
        return df[df["status"] != split]


    def plot_thrend(self, df, plot_name = "plot_thrend.png"):
               

        self._plot_df_xy( df.index, df["count"], plot_name)

        # fig = df.plot(figsize=(13,5), grid=True , marker='o', x_compat=True,legend=True).get_figure()

        # fig.savefig( self.data_path + "/" + plot_name )
    

    def plot_cloudwords_split_text(self, df_m, split_status ):
        df_m['status'] = np.where(df_m.index < self.split_date, 0, 1)
        
        df_split_m = self._split_df( df_m, split_status)
        texts = list(df_split_m["text"])
        if len(texts) > 0:
            self.create_word_cloud(texts, file_name=f"wordCloudTextMentionsFor{self.target_name}_{split_status}.png")

    
    def create_sorted_counts(self, user_names):
        
        # Count for N
        counts = self.get_counts(user_names)
        sorted_counts = self.get_sorted_tuits(counts)
        
        return sorted_counts
    
    def create_top_N_users(self, sorted_tuits, n=21):
            
        # Word Frecuency for N
        df_x = self.create_df_counts(sorted_tuits)
        
        # Plot For N
        topN = df_x.iloc[1:n]
        
        return topN
        
    def get_hashtags(self):
        hashtags = self.get_hashtag_list(self.df_n)
        return hashtags
    
    def get_tuit_frec_ba(self):
        
        # Calculate timediff
        time_diff_b = self.get_timediff(self.df_target, status=0)
        time_diff_a = self.get_timediff(self.df_target, status=1)

        print(f"{self.target_name} Mean Time per tuit before time split: ", time_diff_b.mean())
        print(f"{self.target_name} Mean Time per tuit after time split: ", time_diff_a.mean())
        
        
    def main(self):
        
        # Crrete timed dataframes
        top20 = self.create_top_N_users(self.sorted_counts, n=21)
        
        self.plot_word_count(top20, 
                             titley='Number of Retuits',
                             titlex="Nombres",
                             title="Frecuencia de mensajes")

        #  Cloud Words for N and M
        self.create_word_cloud(self.users_mention,file_name= f"mentionsFrom{self.target_name}.png")
        self.create_word_cloud(self.users_mention_m,file_name= f"mentionsTo{self.target_name}.png")

        #########################################
        
        hashtags = self.get_hashtags()
        
        # Hashtags for N
        self.create_word_cloud(hashtags, file_name= f"hashTagsFrom{self.target_name}.png")

        # Mentions
        
        # Calculate timediff
        if self.df_target is not None:
            self.get_tuit_frec_ba()

        if self.df_target is not None:
                
            ## Plot thrends for target user
            self.plot_df_thrend(self.df_target, status = 0)
            self.plot_df_thrend_per_hour(self.df_target, status=1)

        # Mentions from m
        
        ## Test Split before  M
        ## Objatin tuits mentions
        if self.df_target is not None:
            self.plot_cloudwords_split_text(self.df_m, 1 )
            self.plot_cloudwords_split_text(self.df_m, 0 )

        # Test split after
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Twitter analyser")
    
    required_parser = parser.add_argument_group("required arguments")
    
    required_parser.add_argument("-target_name", "-target", help="target name to analyce", required=True)
    required_parser.add_argument("-doc1", "-d1", help="First document, user tuits", required=True)
    required_parser.add_argument("-doc2", "-d2", help="Second document, user mentions", required=True)

    args = parser.parse_args()

    target_name = args.target_name
    p_1 = args.doc1
    p_2 = args.doc2

    # Create folder for data
    data_path = f"data/{target_name}"
    os.makedirs(data_path, exist_ok=True)

    tw_pro = ProcessTwitter(p_1, p_2,target_name = target_name, data_path=data_path )

    tw_pro.main()