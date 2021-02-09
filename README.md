# DDSP singing experiments
Notebooks and datasets for the Sound and Music Computing 9 semester project

Autumn 2020 - Aalborg University, Copenhagen

#### Please, visit https://juanalonso.github.io/DDSP-singing-experiments/ to listen to the audio examples.

### Main notebooks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/01_train.ipynb)
`01_train`: Notebook used for training the model. It only needs a folder with the sample files and enough time to run. The training process can be interrupted and continued at any point, even if Google closes the connection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/02_run.ipynb)
`02_run`: Notebook used for timbre transfer. It will use the instruments generated with `01_train` to transform the files provided by the user.


### Tools
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/plot_voice_space.ipynb)
`plot_voice_space`: Helper notebook to plot the total loss from all the voice models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/generate_param_space.ipynb)
`generate_param_space`: Helper notebook to train the eva model with different spectral parameters.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/plot_param_space.ipynb)
`plot_param_space`: Helper notebook to plot the total loss from models trained on the eva model with different spectral parameters.

