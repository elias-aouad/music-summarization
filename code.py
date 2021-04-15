import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.io import wavfile
from scipy.signal import resample

import numpy as np

from python_speech_features import mfcc

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from hmmlearn import hmm

from collections import Counter

# import nltk
# from nltk.cluster.kmeans import KMeansClusterer


def plot_segmentation(signal, segments, title, t=20):
    s = signal[:16000*t]
    n = int(s.shape[0] / 16000 / 0.01)
    seg = segments[:n]
    
    colors = mcolors.CSS4_COLORS
    colors = list(colors.keys())
    
    streak_state = {}
    old_label = seg[0]
    for i in range(1, seg.shape[0]):
        new_label = seg[i]
        if new_label == old_label:
            if old_label in streak_state:
                streak_state[old_label]['length'][-1] += 1
            else:
                streak_state[old_label] = {'position':[i], 'length':[2]}
        else:
            old_label = seg[i]
            if new_label not in streak_state:
                streak_state[new_label] = {'position':[i], 'length':[1]}
            else:
                streak_state[new_label]['position'] += [i]
                streak_state[new_label]['length'] += [1]
    
    plt.figure()
    c = 0
    for state in streak_state.keys():
        for i in range(len(streak_state[state]['position'])):
            pos = streak_state[state]['position'][i]
            length = streak_state[state]['length'][i]
            
            i0 = int(pos*0.01*16000)
            i1 = i0 + int(length*0.01*16000)
            
            y = s[i0:i1]
            x = np.linspace(i0/16000, i1/16000, y.shape[0])
            
            try:
                plt.plot(x, y, c=colors[c%len(colors)])
            except:
                pass
        c += 1
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('time (s)')
    plt.show()

def segmentation(features, threshold=0.8):
    segment = [0]
    label = 0
    for t in range(len(features)-1):
        f1 = features[t].reshape(1, -1)
        f2 = features[t+1].reshape(1, -1)
        s = cosine_similarity(f1, f2)
        if s < threshold:
            label += 1
        segment += [label]
    return np.array(segment)

def compute_avg_features(features, segment):
    labels = np.unique(segment)
    avg_features = []
    for label in labels:
        idx = np.where(labels==label)[0]
        segment_features = features[idx]
        avg_features += [np.mean(segment_features, axis=0)]
    return np.array(avg_features)

def merge_similar_segments(avg_features, segments, threshold=0.8):
    new_segments = segments.copy()
    sim_matrix = cosine_similarity(avg_features)

    idx = np.where(sim_matrix > threshold)
    
    i_list = []
    j_list = []
    
    for k in range(len(idx[0])):
        i, j = idx[0][k], idx[1][k]
        if i < j:
            if i in j_list:
                label = i_list[j_list.index(i)]
            else:
                label = i
                i_list.append(i)
                j_list.append(j)
            new_segments[new_segments==i] = label
            new_segments[new_segments==j] = label

    return new_segments

def get_streaks(states, state_i):
    already_in_streak = False
    streak = {}
    idx = None
    for i in range(len(states)):
        if states[i] == state_i:
            if already_in_streak:
                streak[idx] += 1
            else:
                already_in_streak = True
                idx = i
                streak[idx] = 1
        else:
            already_in_streak = False
    return streak

def transform_segments(states):
    output = [states[0]]
    idxs = [0]
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            output += [states[i]]
            idxs += [i]
    return np.array(output), idxs

def count_ngrams(sequence, n):
    n_grams = ['-'.join(sequence[i:i+n].astype(str)) for i in range(len(sequence)-n+1)]
    return Counter(n_grams)

def get_longest_ngram_streaks(states, ngram, n):
    sequence, idxs = transform_segments(states)
    n_grams = np.array(['-'.join(sequence[i:i+n].astype(str)) for i in range(len(sequence)-n+1)])
    
    n_gram_idx = np.where(n_grams == ngram)[0]
    streak = {}
    
    for i in n_gram_idx:
        idx_start = idxs[i]
        if i+n < len(idxs):
            idx_end = idxs[i+n]
        else:
            idx_end = len(states)
        streak[idx_start] = idx_end - idx_start
    longest_ngram_idx = sorted(streak.keys(), key=lambda key:streak[key], reverse=True)[0]
    longest_ngram_streak = streak[longest_ngram_idx]
    
    return longest_ngram_idx, longest_ngram_streak
        

def main():
    # read wav file
    path_to_file = 'all_you_need_is_love.wav'
    original_sample_rate, original_signal = wavfile.read(path_to_file)
    
    # dont want stereo
    signal = 0.5*(original_signal[:, 0]+original_signal[:, 1])
    
    # resample to 16kHz sammple rate
    sample_rate = 16000
    num_resample = int(round(len(signal)*sample_rate/original_sample_rate))
    signal = resample(signal, num_resample)
    
    # take only 100 first seconds
    signal = signal[:int(100*sample_rate)]
    
    # extract MFCC features
    features = mfcc(signal)
    
    # perform segmentation
    segments = segmentation(features)
    
    # compute average feature vector per segment
    avg_features = compute_avg_features(features, segments)
    
    # merge similar segments
    new_segments = merge_similar_segments(avg_features, segments)
    
    # recompute average feature vector per segment
    avg_features = compute_avg_features(features, new_segments)
    
    # K-means for clustering
    kmeans = KMeans(n_clusters=avg_features.shape[0], init=avg_features).fit(features)
    # kmeans_predicted_states = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    
    # training HMM
    n_components = len(set(new_segments))
    hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
    hmm_model.means_prior = kmeans_cluster_centers
    hmm_model.fit(features)
    
    hmm_predicted_states = hmm_model.predict(features)
        
    # output music summary with most represented state
    count_states = Counter(hmm_predicted_states)
    most_represented_state = sorted(count_states.keys(), key=lambda key:count_states[key], reverse=True)[0]
    state_streaks = get_streaks(hmm_predicted_states, most_represented_state)
    
    longest_streak_index = sorted(state_streaks.keys(), key=lambda key:state_streaks[key], reverse=True)[0]
    longest_streak_length = state_streaks[longest_streak_index]
    
    i0 = int(longest_streak_index*0.01*original_sample_rate)
    i1 = i0 + int(longest_streak_length*0.01*original_sample_rate)
    
    summary = original_signal[i0:i1]
    
    wavfile.write(filename='summary.wav', rate=original_sample_rate, data=summary)
    
    # output music summary with most represented n-gram
    n = 10
    sequence_with_no_repetition, _ = transform_segments(hmm_predicted_states)
    count_of_ngrams = count_ngrams(sequence_with_no_repetition, n)
    most_represented_ngram = sorted(count_of_ngrams.keys(), key=lambda key:count_of_ngrams[key], reverse=True)[0]
    longest_ngram_idx, longest_ngram_streak = get_longest_ngram_streaks(hmm_predicted_states, most_represented_ngram, n)
    
    i0 = int(longest_ngram_idx*0.01*original_sample_rate)
    i1 = i0 + int(longest_ngram_streak*0.01*original_sample_rate)
    
    summary_ngram = original_signal[i0:i1]
    
    wavfile.write(filename='summary_ngram.wav', rate=original_sample_rate, data=summary_ngram)

if __name__ == '__main__':
    main()

            
        
