import re


cont_dict = {'didn\'t': 'did not', 'don\'t' : 'do not',}
cont_re = re.compile('(%s)' % '|'.join(cont_dict.keys()))

def expand_cont(s, cont_dict=cont_dict):
    def replace(match):
        return cont_dict[match.group(0)]
    return cont_re.sub(replace, s)

print(expand_cont('you don\'t need didn\'t'))

self.df["tweet"] = self.df["tweet"].apply(lambda x: [item for item in x if item not in stop])

self.df["tweet"] = self.df["tweet"].apply(lambda x: [stemmer.stem(y) for y in x])

if x 