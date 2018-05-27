# common
import pandas as pd
import numpy as np
from pandas import DataFrame
import math

pokemon_data= pd.read_csv('pokemon.csv')

# 変数
bd = 0.6
ng = 0.8
gd = 1.3


#                1   2   3   4   5  #6   7   8   9  11  11  12  13  14  15  16  17  18
type_chart = [[  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, ng, bd,  1,  1, ng,  1],
              [  1, ng, ng,  1, gd, gd,  1,  1,  1,  1,  1, gd, ng,  1, ng,  1, gd,  1],
              [  1, gd, ng,  1, ng,  1,  1,  1, gd,  1,  1,  1, gd,  1, ng,  1,  1,  1],
              [  1,  1, gd, ng, ng,  1,  1,  1, bd, gd,  1,  1,  1,  1, ng,  1,  1,  1],
              [  1, ng, gd,  1, ng,  1,  1, ng, gd, ng,  1, ng, gd,  1, ng,  1, ng,  1],
              [  1, ng, ng,  1, gd, ng,  1,  1, gd, gd,  1,  1,  1,  1, gd,  1, ng,  1],
              [ gd,  1,  1,  1,  1, gd,  1, ng,  1, ng, ng, ng, gd, bd,  1, gd, gd, ng],
              [  1,  1,  1,  1, gd,  1,  1, ng, ng,  1,  1,  1, ng, ng,  1,  1, bd,  2],
              [  1, gd,  1, gd, ng,  1,  1, gd,  1, bd,  1, ng, gd,  1,  1,  1, gd,  1],
              [  1,  1,  1, ng, gd,  1, gd,  1,  1,  1,  1, gd, ng,  1,  1,  1, ng,  1],
              [  1,  1,  1,  1,  1,  1, gd, gd,  1,  1, ng,  1,  1,  1,  1, bd, ng,  1],
              [  1, ng,  1,  1, gd,  1, ng, ng,  1, ng, gd,  1,  1, ng,  1, gd, ng, ng],
              [  1, gd,  1,  1,  1, gd, ng,  1, ng, gd,  1, gd,  1,  1,  1,  1, ng,  1],
              [ bd,  1,  1,  1,  1,  1,  1,  1,  1,  1, gd,  1,  1, gd,  1, ng,  1,  1],
              [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, gd,  1, ng, bd],
              [  1,  1,  1,  1,  1,  1, ng,  1,  1,  1, gd,  1,  1, gd,  1, ng,  1, ng],
              [  1, ng, ng, ng,  1, gd,  1,  1,  1,  1,  1,  1, gd,  1,  1,  1, ng,  2],
              [  1, ng,  1,  1,  1,  1, gd, ng,  1,  1,  1,  1,  1,  1, gd, gd, ng,  1]]

type_dict = {'Normal': 0,
             'Fire': 1,
             'Water': 2,
             'Electric': 3,
             'Grass': 4,
             'Ice': 5,
             'Fighting': 6,
             'Poison': 7,
             'Ground': 8,
             'Flying': 9,
             'Psychic': 10,
             'Bug': 11,
             'Rock': 12,
             'Ghost': 13,
             'Dragon': 14,
             'Dark': 15,
             'Steel': 16,
             'Fairy': 17}

# 1対1でのタイプ相性
def type1(x,y):
    return type_chart[type_dict[x]][type_dict[y]]

# １対2でのタイプ相性(技を出すときなど)
def type2(x,y):
    if y[1] != y[1]:
        return type1(x,y[0])
    return type1(x,y[0]) * type1(x,y[1])

# ポケモン同士としてのタイプ相性(有利な方のタイプの技を繰り出す)
def type22(x,y):
    if x[1] != x[1]:
        return type2(x[0],y)
    return max([type2(x[0],y),type2(x[1],y)])

# 双方向のポケモン同士のタイプ相性
def compatibility(x,y):
    return type22(x,y) * 1.0 /type22(y,x)

def pokemon_to_type(pokemon_num):
    return [pokemon_data.iat[pokemon_num - 1,2], pokemon_data.iat[pokemon_num - 1,3]]

def pokemon_compatibility(x,y):
    return compatibility(pokemon_to_type(x),pokemon_to_type(y))
