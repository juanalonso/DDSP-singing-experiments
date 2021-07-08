# DDSP singing experiments
[Sound and Music Computing](https://www.smc.aau.dk/) - Aalborg University, Copenhagen

#### Please, visit [the audio examples page](https://juanalonso.github.io/DDSP-singing-experiments/) to listen to the results.
#### The paper [Explorations Of Singing Voice Synthesis Using DDSP](https://arxiv.org/pdf/2103.07197) was presented at the 18th Sound and Music Computing conference.

## Main notebooks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/01_train.ipynb)
`01_train`: Notebook used for training the model. It only needs a folder with the sample files and enough time to run. The training process can be interrupted and continued at any point, even if Google closes the connection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/02_run.ipynb)
`02_run`: Notebook used for timbre transfer. It will use the instruments generated with `01_train` to transform the files provided by the user.

## Background
### 👉🏼 (Almost) no ML knowledge required! 👈🏼

**DDSP singing experiments** is built upon the great library [DDSP: Differentiable Digital Signal Processing](https://github.com/magenta/ddsp) by Google's [Magenta team](https://magenta.tensorflow.org/). The library is presented in [this paper](https://arxiv.org/abs/2001.04643) and there is also a great [blog post](https://magenta.tensorflow.org/ddsp) by the original authors.

This work allows us to explore one area of particular interest to us: the creation of tools that facilitate creative experimentation with Deep Neural Networks, while leaving room for serendipity and accidental findings. Applying DDSP to the singing voice has been a consciously daring decision: we wanted to explore the limits of the library by using small datasets extracted from raw, unprepared audio, with no linguistic conditioning.

Machine Learning based singing voice models require large datasets and lengthy training times. **DDSP singing experiments** is a lightweight architecture, based on the DDSP library, that is able to output song-like utterances conditioned only on pitch and amplitude, after 12 hours of training using 15 minutes of unprocessed audio. The results are promising, as both the melody and the singer’s voice are recognizable, and enough leeway exists for further formal research and artistic usage.

You can 1️⃣ read our paper [Latent Space Explorations of Singing Voice Synthesis using DDSP](https://arxiv.org/abs/2103.07197), 2️⃣ dive into the notebooks, where you can easily train and use the models or 3️⃣ [listen to the results](https://juanalonso.github.io/DDSP-singing-experiments/). 


## Design goals

This project has two major goals:

1. **Test the validity of the DDSP architecture to generate a singing voice**. The existing models produce excellent results when they are trained from a clean, high quality monophonic audio source from a single instrument. The problem gets harder if we want to generate singing lyrics: not only the model has to learn the timbre and the transitions between different pitches, it also has to learn about the flow of speech, the relationship between phonemes, the rests, the breath... To make things even more difficult, we want to avoid preprocessing the source audio while keeping it to a minimum in duration. That entails that the model will be trained on a reduced set of pitches, phonemes and transitions.
2. **Create an easy-to-use environment to facilitate model training and timbre transfer to end users.** Machine Learning models are exceedingly complex at three different levels:
	- The structure of the model: the structure is not always clearly defined in the papers.
	- The complexity of the environment: the very specific requirements about libraries, versions, drivers, etc. make difficult setting up the infrastructure.
	- The complexity of the workflow: obtaining the dataset and preparing it for the training process, etc. Even running the training process and getting the results can be difficult.

This complexity acts as a barrier to new DL practitioners or curious programmers that want to get familiar with something different than the simplest examples. To lower this barrier, we have followed two principles when designing these notebooks: *easy to use* and *easy to reuse*. Easy to use means that after a minimal configuration (setting up three folders) the user is ready to start training or producing audio. Easy to reuse means that the system is aware of previous operations, so the dataset is generated only once, the training can be interrupted at any point and it will be restored in the next run, and models and audio examples are only reloaded on demand, speeding up the process.

## Development notes

To achieve the design goals, we provide a series of Colab notebooks. Colab provides a virtual machine per session, with a GPU and a predetermined amount of RAM and disk space. Colab sessions last up to twelve hours. Upon an unexpected disconnection, a user may lose all the data stored in the virtual machine. For that reason, it is fundamental to save the final or temporal results to another drive. In this project, we use Google Drive as permanent storage space. All the required data are copied to Colab when the session starts, and the results will be stored in Drive, so no data is lost in case of disconnection.

### Folder structure

To facilitate access to the data, all the notebooks expect to find a similar folder structure in Google Drive, so all data is shared without needing to move it around. The base folder is defined at the top of each notebook. Inside this folder three folders are needed: `audio`, where the audio files, temporal checkpoints and datasets are stored; `examples`, which contains the files we are going to present to the model to modify their original timbre; and `instruments`, where the trained models are stored in zip format.

<img width="60%" alt="fig_folderstructure" src="https://user-images.githubusercontent.com/1846199/110327534-6364fd80-801a-11eb-9e81-2455f0cfee11.png">

Managing files in Google Drive can be a nightmare if done via the standard web interface. It is very recommended to use [Google Drive for desktop](https://support.google.com/drive/answer/7329379), an official free utility that allows the user to manage files and folders in Google Drive using the user's computer's native interface.

## Training the model with `01_train`

### Data preparation

For each instrument we want the system to learn its timbral characteristics, we need to create a folder inside the `audio` folder and place there the source audio files in wav or mp3 format. We will use `newinst` as the folder / instrument name for the rest of the section. No additional conversions (bit-depth, sample frequency, number of channels) are needed. Splitting the audio files into 3-minute chunks is recommended.

### Notebook configuration

All the configuration values are entered in the first two cells of the notebook. The first one mounts the Google Drive file system and prompts for an authorization code. The second cell defines 1) the entry point and 2), the name of the folder with the source audio files.

Also, the runtime must be changed to GPU to take advantage of the accelerated hardware. (Choose `Runtime > Change runtime type > GPU` in the Colab menu)

### Training the model

Once the notebook is set up, the rest of the process is automatic, and the training starts when we execute the whole notebook (`Runtime > Execute all` in the Colab menu). The notebook will download the DDSP library, import the required python libraries, create additional folders to store the checkpoints and the final instrument, and then will create the dataset from the audio files.

If the dataset already exists (by checking `audio\newinst_dataset`) it will skip this step and copy the existing dataset into Colab temporal storage. Otherwise, the dataset is created by executing `ddsp_prepare_tfrecord`, which reads all the audio files in the `audio\newinst` folder, resamples them at 16kHz and splits them into four-second chunks with one-second hops. For each chunk, the system takes 250 timeframes per second and computes on each frame the loudness in dB, f0 and the confidence of the estimation. The newly created dataset is stored both in the Colab temporal storage space and in Drive, for safekeeping in the `audio\newinst_dataset` folder. Also, two additional files are created:

1. The pickle file (`dataset_statistics.pkl`) with loudness and pitch statistics that will be used to preprocess the audio in the second notebook, and
2. a configuration file (`operative_config-0.gin`) with the full set of parameters needed to define, train and use the model.

Once the dataset is available, the notebook picks an element from the dataset and displays its spectrogram, the f0 estimation and confidence values, the loudness, and an audio player, so we can check for errors.

Then, the notebook launches Tensorboard, so we can visualize the total loss and the steps per second. By default, Tensorboard graphs are not automatically updated, so we will need to click on the refresh icon –or change the default configuration– to redraw the graphs with the latest scalar values. The complete Tensorboard logs are stored in the folder// `audio\newinst_checkpoints\summaries\train`, so they are preserved between different runs.

To train the model (from scratch or from the latest checkpoint), the `ddsp_run` command is executed. For this particular case, we are using a customized configuration file which tells the system not to learn the reverb of the source audio. The configuration file is a simplified version of the original `solo_instrument.gin` and it is available [in the GitHub repository](https://github.com/juanalonso/DDSP-singing-experiments/tree/main/gin/models).

The system will train for 40k steps, with a batch size of 32, saving a checkpoint to Drive every 250 steps, and keeping the last 5 generated checkpoint files in the checkpoints folder (`audio\newinst_checkpoints` in our example).  In the case we get disconnected, executing again all the cells will let the notebook recover gracefully from the last saved checkpoint.

Once the training has finished or is interrupted, the notebook will run the model on an element of the dataset and will present side by side both the original and reconstructed spectrogram and audio. This comparison, together with the Tensorboard, should give us an intuition about the quality of the model: usually, a loss value over 6 means there is room for improvement, and a value under 5 points to overfitting.

The last step is creating the standalone model / instrument file. This file will be used in the timbre transfer notebook and it is a zip file with the most recent checkpoint, the configuration file and the pickle file. The file is copied to the `instruments` folder (in our example, it will be `instruments\newinst.zip`).

### Tips and Tricks

* To create the dataset, it is better to split the source audio into several shorter audio files (up to three minutes) instead of using a single longer file. In our experience, longer files tend to cause out-of-memory errors.
* It is difficult to estimate the duration of the training process. The GPU assignation method is unknown to the user, and the time per step also varies during a session. As a rule of thumb, we use a conservative estimation of 3000 steps per hour, roughly equivalent to 0.8 steps per second.
* Checkpoint files for this model are about 58MB. It is very easy to run out of Drive storage space when training several instruments and keeping old unused checkpoints.
* To be able to keep training the model, do not delete the checkpoints folder, otherwise the training will start from scratch. It is also recommended to keep the dataset folder. If not present, the dataset will be recreated, and it is a slow operation.
* The instrument file should be around 50-55MB in size. If it is bigger, it means that more than a set of checkpoints are stored inside, usually because the neural network has been trained more than once in the same session. This can cause a problem when using the instrument file in the timbre transfer notebook, because the system will pick one of the checkpoint files at random. If this happens, we can manually delete the undesired checkpoints from the zip file.

## Timbre transfer with `02_run`

### Data preparation

In this section we will use the folder `instruments`, where the models are automatically stored, and the folder `examples` where we will place the source audio files (in wav and mp3 format) we want to transform.

### Notebook configuration

Similar to the training notebook, the first cell mounts the Google Drive file system and prompts for an authorization code. The second cell defines 1) the entry point and 2) the name of the folder with the instruments and the name of the folder with the examples, `instruments` and `examples` by default.

### Running the model

This notebook is interactive, and possesses a minimal GUI to load instruments, examples and fine-tune the output. When the notebook is executed, it will download the required libraries, and copy the examples and instruments from Drive to Colab.

The first step is choosing the instrument and the example.

<img width="60%" alt="Choosing the instrument and the example" src="https://user-images.githubusercontent.com/1846199/110327540-64962a80-801a-11eb-8a0c-f63265215f62.png">

Selecting one instrument will unzip the instrument file, load the configuration file, the model and the pickle file. Selecting one example the notebook will load the example, and extract the f0, confidence and loudness. Then, the model is restored. Computations are minimized, so choosing another example does not affect the current instrument and choosing another instrument does not affect the current example.

Before running the model, we may need to preprocess the example audio.

<img width="30%" alt="preprocess the example audio" src="https://user-images.githubusercontent.com/1846199/110327544-652ec100-801a-11eb-8304-37eed33edbb6.png">

The idea behind preprocessing the original audio is to make it more similar to the audio the model has been trained on (loudness and pitch), which renders a more faithful reconstruction. The parameters we can configure are:

* **Show full output**: This control is not strictly preprocessing: if this checkbox is checked, the output will also show a player for the original audio and the original spectrogram.
* **Use loudness statistics**: If checked, the preprocessor will use the data in the pickle file to improve the results by adjusting the loudness of the original audio to better match the training data using quantile normalization.
* **Mask threshold**: The mask is computed based on the note-on ratio, a function of the loudness and the f0 confidence. It is used to attenuate the parts of the source audio where CREPE returns a low confidence on the pitch and / or the volume is low. The higher the mask threshold is, the more parts it will attenuate. This control will only be considered if the "Use loudness statistics" checkbox is checked.
* **Attenuation**: This value sets how much the loudness is attenuated in the places masked out. This control will only be considered if the "Use loudness statistics" checkbox is checked.
* ***Autotune***: This value readjusts the f0 estimation, 'snapping' the values to the nearest semitone. 0 means no change, 1 means full pitch quantization.
* **Octave shift**: If the original instrument is trained in a different pitch range than the example we want to process, we can transpose the example any number of octaves (-2 to 2 is recommended), so the example audio matches the instrument range. For example, when running a female voice (example) through a male voice model (instrument), results are usually improved if we transpose the example -1 octave.
* **Loudness shift**: This control allows the modification of the example loudness when the loudness is very different between the example and the instrument. By adjusting the gain, we will get more natural results.


The model is run by pressing the "Transfer timbre" button. The results will appear below, and they are not cleared automatically between runs,so we can execute several experiments and compare the results easily.

The output presented by the model is (from top to bottom):

* Audio player and spectrogram of the original audio. Only if "Show full output" is checked.
* Audio player and spectrogram of the synthesized audio.
* Graph showing the loudness of the example, before (Original) and after (Norm) preprocessing the amplitude with the loudness statistics.
* Graph showing the pitch of the example as computed by CREPE, the mean pitch from the instrument and from the example, and the autotuned pitch. Comparing mean pitches in this graph is the fastest way to estimate the value of the control "Octave shift"
* Plot of the f0 confidence, as computed by CREPE.
* Graph showing the note-on ratio, the mask threshold and the mask. Note that the mask height represents nothing, as it has only two values, True or False.

<img width="60%" alt="output presented by the model" src="https://user-images.githubusercontent.com/1846199/110327546-65c75780-801a-11eb-9c97-dc9a3103e9e6.jpg">



### Additional Tools
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/plot_voice_space.ipynb)
`plot_voice_space`: Helper notebook to plot the total loss from all the voice models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/generate_param_space.ipynb)
`generate_param_space`: Helper notebook to train the eva model with different spectral parameters.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/DDSP-singing-experiments/blob/main/tools/plot_param_space.ipynb)
`plot_param_space`: Helper notebook to plot the total loss from models trained on the eva model with different spectral parameters.

### Citation

If you use this code please cite it as:

```latex
@inproceedings{
  alonso2021latent,
  title={Latent Space Explorations of Singing Voice Synthesis using DDSP},
  author={Alonso, Juan and Erkut, Cumhur},
  booktitle={Proceedings of 18th Sound and Music Computing Conference},
  year={2021}
}
```
