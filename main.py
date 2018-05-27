# common
import pandas as pd
import numpy as np
from pandas import DataFrame

import type_compatibility as tc

pokemon_data= pd.read_csv('pokemon.csv')
buttle_data = pd.read_csv('train.csv')

list = []
for i in buttle_data.index:
    list.append(tc.pokemon_compatibility(buttle_data.First_pokemon[i],buttle_data.Second_pokemon[i]))

buttle_data['Type_compatibility'] = list

First_HP = []
First_Attack = []
First_Defense = []
First_Sp_Attack = []
First_Sp_Defense = []
First_Speed = []

for i in buttle_data.index:
    First_HP.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,4])
    First_Attack.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,5])
    First_Defense.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,6])
    First_Sp_Attack.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,7])
    First_Sp_Defense.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,8])
    First_Speed.append(pokemon_data.iat[buttle_data['First_pokemon'][i] - 1,9])

buttle_data['First_HP'] = First_HP
buttle_data['First_Attack'] = First_Attack
buttle_data['First_Defense'] = First_Defense
buttle_data['First_Sp_Attack'] = First_Sp_Attack
buttle_data['First_Sp_Defense'] = First_Sp_Defense
buttle_data['First_Speed'] = First_Speed

Second_HP = []
Second_Attack = []
Second_Defense = []
Second_Sp_Attack = []
Second_Sp_Defense = []
Second_Speed = []

for i in buttle_data.index:
    Second_HP.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,4])
    Second_Attack.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,5])
    Second_Defense.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,6])
    Second_Sp_Attack.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,7])
    Second_Sp_Defense.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,8])
    Second_Speed.append(pokemon_data.iat[buttle_data['Second_pokemon'][i] - 1,9])

buttle_data['Second_HP'] = Second_HP
buttle_data['Second_Attack'] = Second_Attack
buttle_data['Second_Defense'] = Second_Defense
buttle_data['Second_Sp_Attack'] = Second_Sp_Attack
buttle_data['Second_Sp_Defense'] = Second_Sp_Defense
buttle_data['Second_Speed'] = Second_Speed

buttle_data['Win?'] = (buttle_data['First_pokemon'] == buttle_data['Winner'])
buttle_data['Win?'] = buttle_data["Win?"].apply( lambda x: 1 if x else 0)

buttle_data = buttle_data.drop(['id','First_pokemon','Second_pokemon','Winner'], axis=1)

buttle_data.to_csv("buttle_data.csv")
