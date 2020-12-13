from collections import OrderedDict
import gc
import pandas as pd
import re
from typing import Iterable

# cleans up most punctuation, removes accents
# replace by space to avoid word concatenation
#     whitespace splitting then removes added spaces 
REMOVE_CHAR = OrderedDict({
'{':" ", '}':" ", ',':"", '.':" ", '!':" ", '\\':" ", '/':" ", '$':" ", '%':" ",
'^':" ", '?':" ", '\'':" ", '"':" ", '(':" ", ')':" ", '*':" ", '+':" ", '-':" ",
'=':" ", ':':" ", ';':" ", ']':" ", '[':" ", '`':" ", '~':" ",
})

# TODO (mranger): Unit test text cleaning procedure more rigorously
#   Add dict to remove accents? Also smart rules on some characters
#   eg. "Award-Winning" -> "award winning" but "f-150"->"f150" (or even "f-150")
#
#   other case: numbers with commas "12,334" -> "12334" (this is mandatory)
#
#   also verify perf of calling lower() -> replace dict 
#   vs adding upper->lower to replaceDict regex (single regex pass)

def tokenizeCol(toClean: Iterable[str],
                split: [bool, str]=True,
                nanValue='',
                lower: bool=False,
                replaceDict: OrderedDict=REMOVE_CHAR,
                expand=True
                ) -> pd.Series:
    """
    applies transormatiosn to a series of strings in a vectorized fashion

    Splitting is applied last. Order is:
        replace NaN -> lower -> sub string mapping -> splitting

    :param toClean: the input vector to process
    :param lower: 
        whether to lowercase the input
    :param split: 
        whether to split the input
        default is True, which splits on whitespace
        can also split on str/regex if eg. split="myregex|thing"
        acts like python str.split() input
    :param nanValue:
        What to replace NaN or empty strings with. Default is empty string
    :param replaceDict: 
        a dictionary of str->str to map subwords is applied after lowering, 
        but before splitting. Note that token keys input should probably NOT
        be a regex, since keys and values get compiled to a single regex 
        delimited by pipes (eg. "a|b|c|d")
        before being applied -- user beware if doing this.

        Also note that REGEX ENGINES ARE EAGER in evaluation: replaceDict is an
        ORDERED dict and the first match in the dict will replace the first match
        in the string
    :type toClean: pd.Series[str]
    :type lower: bool
    :type split: [bool, str]
    :type replaceDict: dict[str->str]

    :return: cleaned vector of vectors strings in a pd.Series
    :rtype: pd.Series of list[str]
    """
    result = None
    if type(toClean) == pd.Series:
        result = toClean.copy()
    else:
        result = pd.Series(toClean)
    if nanValue is not None:
        result.loc[(result.isna()) 
                   | (result.str.len() < 1)] = nanValue
    if lower:
        result = result.str.lower()
    if replaceDict:
        if type(replaceDict) is not dict:
            replaceDict = OrderedDict(replaceDict)
        rep = dict((re.escape(k), v) for k, v in replaceDict.items())
        # regex to change the strings
        pattern = re.compile("|".join(rep.keys()))
        def replacer(text):
            return rep[re.escape(text.group(0))]
        result = result.str.replace(pattern, replacer)
    if type(split) is str:
        result = result.str.split(split, expand=expand)
    elif split:
        result = result.str.split(expand=expand)
    gc.collect()
    return result


def tokenize(colList: Iterable[Iterable[str]],
             split: [bool, str]=True,
             nanValue='',
             lower: bool=False,
             replaceDict: OrderedDict=REMOVE_CHAR,
             expand=True
             ) -> pd.Series:
    """
    Applies cleaning and tokenizing to one or many columns where each column is 
    a series of strings. Concatenates the results into a series of lists of tokens.

    Splitting is applied last. Order is:
        replace NaN -> lower -> sub string mapping -> splitting

    :param colList: 
        Either an iterable of strings, or an iterable of iterables of strings.
        This can be a pandas dataframe, a list of lists, a list of pd.Series, etc.
    :param lower: 
        whether to lowercase the input
    :param split: 
        whether to split the input
        default is True, which splits on whitespace
        can also split on str/regex if eg. split="myregex|thing"
        acts like python str.split() input
    :param nanValue:
        What to replace NaN or empty strings with. Default is empty string
    :param replaceDict: 
        a dictionary of str->str to map subwords is applied after lowering, 
        but before splitting. Note that token keys input should probably NOT
        be a regex, since keys and values get compiled to a single regex 
        delimited by pipes (eg. "a|b|c|d")
        before being applied -- user beware if doing this.

        Also note that REGEX ENGINES ARE EAGER in evaluation: replaceDict is an
        ORDERED dict and the first match in the dict will replace the first match
        in the string
    :type toClean: pd.Series[str]
    :type lower: bool
    :type split: [bool, str]
    :type replaceDict: dict[str->str]

    :return: cleaned vector of vectors strings in a pd.Series
    :rtype: pd.Series of list[str]
    """
    # if pd.DataFrame, make into list of string columns
    if type(colList) is pd.DataFrame:
        # this should just be mapping ptrs, not copying data (efficient)
        colList = [colList[c] for c in colList]
    # Detect single column input
    try:
        if type(colList[0]) is str:
            return tokenizeCol(colList, split=split, nanValue=nanValue, 
                               lower=lower, expand=expand, 
                               replaceDict=replaceDict)
    except KeyError: # pandas indexing breaks on [0] often
        if type(colList.iloc[0]) is str:
            return tokenizeCol(colList, split=split, nanValue=nanValue, 
                               lower=lower, expand=expand, 
                               replaceDict=replaceDict)
    result = tokenizeCol(colList[0], split=split, nanValue=nanValue, 
                         lower=lower, expand=expand, replaceDict=replaceDict)
    for col in colList[1:]:
        result = result + tokenizeCol(col, split=split, nanValue=nanValue, 
                                      lower=lower, expand=expand, 
                                      replaceDict=replaceDict)
    return result
