from wordcloud import WordCloud
from konlpy.tag import Twitter
from collections import Counter


def get_wordcloud_list(result):
    counts = Counter(result)
    tags = counts.most_common(40)
    return counts


def create_wordcloud(tags):
    wc = WordCloud(font_path='./data/NanumGothic.otf',
                   background_color='white',
                   width=512, height=512,
                   max_font_size=500,
                   max_words=1000)

    cloud = wc.generate_from_frequencies(tags)

    # 생성된 WordCloud를 test.jpg로 보낸다.
    cloud.to_file('test.jpg')
