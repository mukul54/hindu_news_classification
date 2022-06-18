# README
## File Structure
- `assets`: contains all the datasets and different plots
- `data_pickles`: Datasets split vectors stored as pickle files
- `news`: virtual Environment
- `notebooks`: ALl the training notebooks also contains notebook for scraping
- `report`: Files related to report submission of the project
- `src`: Major folder
- `src/models`: contains trained model as pickle format
- `src/scrapping`: Scripts related to scrapping a website
- `src/static`: static file related to web app e.g images or css files
- `src/templates`: html file to run the web apps

- `src/clean_text.py`: Scripts for raw text cleaning
- `src/features.py`: Script for tfidf feature generation
- `src/train_classifier`: look at the notebook for training the model. This script is incomplete right now.
- `src/visualize.py`: scripts for plotting different graphs

## Running the Code

- Install all the requirements using `requirements.txt` file.
- change directory to `src`
- run `python3 app.py`
