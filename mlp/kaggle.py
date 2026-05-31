import kagglehub

def gtzan(path_download):
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification", path=str(path_download))
    print("Path to dataset files:", path)

def neuralaudio(path_download):
    path = kagglehub.dataset_download("mimbres/neural-audio-fingerprint", path=str(path_download))
    print("Path to dataset files:", path)

def spotify3sec(path_download):
    path = kagglehub.dataset_download("mrodriguez2/spotify-3-second-mel-spectrograms", path=str(path_download))
    print("Path to dataset files:", path)

def fma(path_download):
    path = kagglehub.dataset_download("imsparsh/fma-free-music-archive-small-medium", path=str(path_download))
    print("Path to dataset files:", path)

#path = kagglehub.dataset_download("iamsouravbanerjee/music-genre-classification-dataset-fma")

# gtzan(./datasets/gtzan-data)
