"""NLP_Project.ipynb
Using the [psaw library](https://github.com/dmarx/psaw) to fetch comments from [pushshift.io api](https://github.com/pushshift/api)
"""
#!pip install psaw
import pdb
import pandas as pd
import datetime as dt
import os
import sys
from tqdm import tqdm
from psaw import PushshiftAPI

class Downloader:
  save_local = True
  max_comment_count = 100000
  filter_fields=["body","score","subreddit"]
  api = None
  
  def __init__(self, save_local = True, outpath = None, max_comment_count = 100000, compress=False, filter_fields = []):
    self.save_local = save_local
    self.max_comment_count = max_comment_count
    self.api = PushshiftAPI(rate_limit_per_minute = 60, max_results_per_request = 100)
    self.outpath = outpath
    self.compress = compress
    self.filter_fields = filter_fields

  """
  grabs the subreddit and returns the comments as a dataframe
  
  TODO: update code to allow for before and after range
  """
  def fetch_subreddit(self, content_type, content_name):
    before_epoch=int(dt.datetime.now().timestamp())
    content = []
    
    # rate limit = 1 per second, default
    # comment limit = 100 per request, default

    # check for filter
    if len(self.filter_fields) > 0:
      # if we want comments
      if content_type == "comments":
        gen = self.api.search_comments(
            subreddit=content_name,
            filter=self.filter_fields,
            before=before_epoch
        )
      elif content_type == "submissions": # otherwise we want submissions
        gen = self.api.search_submissions(
            subreddit=content_name,
            filter=self.filter_fields,
            before=before_epoch,
            score='>2',
            is_self=True # only want selftext submissions
        )
      elif content_type == "user": # otherwise we want user comments
        gen = self.api.search_comments(
            filter=self.filter_fields,
            before=before_epoch,
            author=content_name
        )
      else:
        raise Exception("Content type not valid")
    else:
      # if we want comments
      if content_type == "comments":
        gen = self.api.search_comments(
            subreddit=content_name,
            before=before_epoch
        )
      elif content_type == "submissions": # otherwise we want submissions
        gen = self.api.search_submissions(
            subreddit=content_name,
            before=before_epoch,
            score='>2',
            is_self=True # only want selftext submissions
        )
      elif content_type == "user": # otherwise we want user comments
        gen = self.api.search_comments(
            before=before_epoch,
            author=content_name
        )
      else:
        raise Exception("Content type not valid")

    with tqdm(total=self.max_comment_count) as pbar:
      for c in gen:
        content.append(c)
        pbar.update(1)
        if len(content) >= self.max_comment_count:
          break

    if content_type != "user":
      df_subreddit = pd.DataFrame([thing.d_ for thing in content])#.drop(["created_utc", "created"], axis=1)
      if content_type == "comments":
        df_subreddit = df_subreddit[~(df_subreddit.body.isin(["[removed]","[deleted]"])) & (df_subreddit.score > 0)]
      else:
        df_subreddit = df_subreddit[~(df_subreddit.selftext.isin(["[removed]","[deleted]"])) & (df_subreddit.score > 0) & (df_subreddit.num_comments > 1)]
    else:
      df_subreddit = pd.DataFrame([thing.d_ for thing in content])

    """Pickle the data for download if necessary"""
    if self.save_local:
      filename = content_name + "_" + content_type + ".csv"
      if self.compress:
        compression_opts = dict(method='zip', archive_name=filename)
        df_subreddit.to_csv(os.path.join(self.outpath, filename), index=False, compression=compression_opts) 
      else:
        df_subreddit.to_csv(os.path.join(self.outpath, filename), index=False)
    return df_subreddit

  """static loader of pickled subreddit"""
  def load_pickled_sub(subreddit_path):
    return pd.read_pickle(subreddit_path)

# python Downloader.py outpath comment_count type subreddits
# ex: python Downloader.py "./data" 10000 submission liberal conservative
def main(argv):
  dloader = Downloader(outpath = argv[0], max_comment_count = int(argv[1]))
  for subreddit in argv[3:]:
    dloader.fetch_subreddit(argv[2], subreddit)

if __name__ == "__main__":
    main(sys.argv[1:])