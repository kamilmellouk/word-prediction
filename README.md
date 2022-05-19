# Word prediction

## About the project

This project assignment was was implemented by group 4 for the course [DD2417 - Language Engineering](https://www.kth.se/student/kurser/kurs/DD2417?l=en) at [KTH Royal Institute of Technology](kth.se).

It consists of a word prediction software, using neural networks and n-gram modeling.


<p align="right">(<a href="#top">back to top</a>)</p>

## Setup

First clone the repository
```
git clone https://github.com/medkamilmel/word-prediction.git
```

Install the requirements
```
pip3 install -r requirements.txt
```

Download the data, here are the two datasets we used:

* [Blogposts dataset](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip)
* [NewsCrawl 2010 from WMT 2014 Shared Task](https://login.microsoftonline.com/3db27ecc-1791-4dda-9b51-798adfa4a3ca/oauth2/authorize?client_id=00000003-0000-0ff1-ce00-000000000000&response_mode=form_post&protectedtoken=true&response_type=code%20id_token&resource=00000003-0000-0ff1-ce00-000000000000&scope=openid&nonce=257FCB20008A201134AC3CEEE825F8290A7EC230C9852A75-C83996C153AB845A0032038C7A3D6C72A0148C28CB12FD622FBFCFF6FDD8537B&redirect_uri=https%3A%2F%2Fkth-my.sharepoint.com%2F_forms%2Fdefault.aspx&state=OD0w&claims=%7B%22id_token%22%3A%7B%22xms_cc%22%3A%7B%22values%22%3A%5B%22CP1%22%5D%7D%7D%7D&wsucxt=1&cobrandid=11bd8083-87e0-41b5-bb78-0bc43c8a8e8a&client-request-id=bb043fa0-20d3-4000-28ea-ce13ee8006a3) (KTH acess only)

<p align="right">(<a href="#top">back to top</a>)</p>

## How to run

### Neural networks

Inside the `nn/` folder, run
`python3 main.py`
and you will be able to start typing words/sentences. Whenever you press enter, the program will output 3 predictions for the current word, or the next one if your input ends with a whitespace. Input exit if you wish to end the program.

### Trigram model

Inside the `trigram/` folder, run to following to build a trigram model file from the raw data:
```
python3 TrigramTrainer.py -f path/to/data -d path/to/model
```
Optional: add the `-ls` flag to apply laplace smoothing and `-lc` to lowerise the data when processing it.

Inside the `trigram/` folder, run to following to run the prediction CLI from a model file:
```
python3 TrigramPredictor.py -m path/to/model
```
Optional: add `-k path/to/test_data` to compute the proportion of saved keystrokes when running the predictor on test data. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Project structure

* `nn/` contains the neural network predictor
* `trigram/` contains the trigram predictor

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Axel Larsson - [axlarss@kth.se](mailto:axlarss@kth.se)

Kamil Mellouk - [mellouk@kth.se](mailto:mellouk@kth.se)

<p align="right">(<a href="#top">back to top</a>)</p>