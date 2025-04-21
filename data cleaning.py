import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import missingno as msno
from collections import defaultdict
import zipcodes
from pprint import pprint
import requests
import pandas as pd
import io
import csv
from difflib import get_close_matches,SequenceMatcher
from IPython.display import display

"""
try to normalize data cleaning and evaluate data quility
"""
food_dataset = pd.read_csv("Food_Inspections_20250216.csv")
