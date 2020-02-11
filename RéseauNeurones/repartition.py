import split_folders

# Cette librairie permet de facilement découper notre dossier src en 3 dossiers train / test / val
split_folders.ratio('./data/src', output="./data/", seed=1337, ratio=(.5, .2, .3))