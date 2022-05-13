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

<p align="right">(<a href="#top">back to top</a>)</p>

## How to run

### Neural networks

Inside the `nn/` folder, run
`python3 main.py`
and you will be able to start typing words/sentences. Whenever you press enter, the program will output 3 predictions for the current word, or the next one if your input ends with a whitespace. Input exit if you wish to end the program.

### Trigram model

NOT FINISHED YET

<p align="right">(<a href="#top">back to top</a>)</p>

## Project structure

* `nn/` contains the neural network predictor
* `trigram/` contains the trigram predictor

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Axel Larsson - [axlarss@kth.se](mailto:axlarss@kth.se)

Kamil Mellouk - [mellouk@kth.se](mailto:mellouk@kth.se)

<p align="right">(<a href="#top">back to top</a>)</p>