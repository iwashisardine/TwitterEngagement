import tweepy
import csv
import datetime
from pytz import timezone
import os
from dotenv import load_dotenv
import csv
from tqdm import tqdm

load_dotenv()

#Twitter API KEY
Consumer_key = os.environ['Consumer_key']
Consumer_secret = os.environ['Consumer_secret']

#認証
auth = tweepy.AppAuthHandler(Consumer_key, Consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#検索キーワード設定
keyword = 'わたし'
banwords = ['#質問箱']

#option等
option = ' exclude:replies exclude:retweets'
query = keyword + option
for word in banwords:
    query = query + ' -' + word

#タイムゾーンの指定 UTC => JST
def utc_jst(utc_time):
    jst_time = utc_time.astimezone(timezone("Asia/Tokyo"))
    str_time = jst_time.strftime("%Y-%m-%d_%H:%M:%S")
    return str_time

start_time = datetime.datetime.now().astimezone(timezone('Asia/Tokyo'))
yesterday_time = start_time + datetime.timedelta(days=-1)
until_time = yesterday_time.strftime("%Y-%m-%d_%H:%M:%S_JST")

#出力csvにheaderを付加
header = [
    'engagement', 'text', 'time', 'follow', 'follower', 'liked', 'RT', 'get'
]

file = './data/watasi_tweet.csv'
with open('/content/drive/MyDrive/data/' + file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

#Tweetの収集
for tweet in tqdm(
        tweepy.Cursor(api.search,
                      q=query,
                      until=until_time,
                      tweet_mode='extended',
                      result_type='mixed',
                      count=100).items()):
    if tweet.favorite_count != 0 and tweet.retweet_count != 0:
        datum = [(tweet.favorite_count + tweet.retweet_count), tweet.full_text,
                 utc_jst(tweet.created_at), tweet.user.friends_count,
                 tweet.user.followers_count, tweet.favorite_count,
                 tweet.retweet_count,
                 utc_jst(datetime.datetime.now())]

        with open('/content/drive/MyDrive/data/' + file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(datum)