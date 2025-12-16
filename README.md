# A Regression Model to Detect CrossTalk in Microscopy Images

## Setup

### Step 1: Install a Python Distribution

We recommend using conda as it's relatively straightforward and makes the management of different Python environments simple. You can install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) (miniconda will suffice).

## Step 2: Set Up Environment

Once conda is installed, open a terminal (Mac) or AnaConda Prompt (Windows) and run the following series of commands:

```
conda create --name crosstalk-detection python=3.13
conda activate crosstalk-detection
python -m pip install -r <path to this repo>/requirements.txt
```
where you need to replace `<path to this repo>` with the location on your file system where you downloaded this repo. You will be presented with a list of packages to be downloaded and installed. The following prompt will appear:
```
Proceed ([y]/n)?
```
Hit Enter and all necessary packages will be downloaded and installed - this may take some time. When complete, you can deactivate the environment you have created with the following command.

```
conda deactivate
```
You have successfully set up your environment!


## Training

## Testing
