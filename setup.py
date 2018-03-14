from setuptools import setup

setup(name='TSA',
      version='0.1',
      description='Graduate work for CGI on Twitter Sentiment analysis',
      url='https://github.com/Brew8it/Twitter-SA-Project',
      author='Brew8it and Selberget',
      author_email='bandgren2@gmail.com & johan.selberg@gmail.com',
      license='MIT',
      packages=['TSA'],
      install_requires=[
                        'NumPy',
                        'SciPy',
                        'scikit-learn',
                        'pandas',
                        'nltk',
                        'tweepy',
                        'bs4',
                        'flask',

      ],
      zip_safe=False)