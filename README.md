

# Twitter analycer



## Use

Install:
https://github.com/taspinar/twitterscraper


**Get the Data**

```
twitterscraper Trump -l 1000 -bd 2017-01-01 -ed 2017-06-01 -o tweets.json
```

```
twitterscraper Trump --limit 1000 --output=tweets.json
```


```
twitterscraper Trump -l 1000 -o tweets.json
```

**Most used**

```
twitterscraper userName --user -o userName.json

```


```
twitterscraper "userName to:userName" -o userNameMentions.json
```


**Use the Script**

```
python main.py -d1 inputData/userName.json -d2 inputData/userNameMentions.json -target_name "User Name"
```
