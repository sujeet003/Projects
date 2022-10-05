from flask import Flask, request, render_template
import pickle
import numpy as np

app=Flask(__name__,template_folder='./templates')

model = pickle.load(open("model.pkl", "rb"))

print(model.feature_importances_)
print(model.get_booster().feature_names)

'''
[0.07299763 0.07738809 0.07496335 0.08530349 0.08100301 0.08464274
 0.0781937  0.0732685  0.08343643 0.07289273 0.07621971 0.06348531
 0.07620519]
['song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']
'''