import os
import langid
import pandas as pd
import collections


def language_detection(text):
    """Detect language of content

    Args:
        text (unicode): Input text

    Returns:
        str: Language"""
    if text is None:
        return None
    lang, value = langid.classify(text)
    return lang


if __name__ == "__main__":
    path = os.path.realpath('') + '/'
    print('Reading in data file...')
    data = pd.read_csv(path + 'Sentiment Analysis Dataset.csv',
                       usecols=['Sentiment', 'SentimentText'],
                       error_bad_lines=False)

    print('Pre-processing tweet text...')
    corpus = data.loc[:10000, 'SentimentText']

    language = corpus.apply(language_detection)
    s_languages = pd.Series(dict(collections.Counter(language)))
    s_languages.plot.pie()
# %%
