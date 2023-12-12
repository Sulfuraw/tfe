# Conception of an AI for the game Stratego: A guided Monte-Carlo Tree Search solution

This repository is dedicated to my master thesis about the production of an AI using the MCTS (Monte-Carlo Tree Search) algorithm. The game playing is done on pyspiel game support (https://github.com/google-deepmind/open_spiel)

See the Paper.pdf for more information.

This was done by Thomas Robert.

## Requirement

Python and the following python package:
- numpy
- pandas
- matplotlib
- pickle
- curses

And also obviously the packages **open_spiel** and **pyspiel**. But the version needed for both is not the official one, so I included it in the repo. You need to add these file to the PATH so that python recognize the imports. (Official one available there: https://github.com/google-deepmind/open_spiel)

## Additional information

The TODOs left in the code are some possible improvements that could be tested.

**script**: Python file to compile the information of the game statistics into more readable data. Markdown of the bots statistics.

All the following files are inside the **Src** folder:

**main**: Python file to launch all the matches and training that we could launch, it contains many use case. It also contains the different loging and game statistics.

**statework**: Python file for all function that can be applied to a state, or to a game in general, so that multiple bot can use these. Many helper function.

**customBot**: The bot that I created using mcts with some modifications.

**mctsBot**: Basic mcts agent in purposing of comparison.

**basicAIBot**: Temp Bot that plays randomly but the goal was trying to copy the structure used from the bot of the competion below (asmodeus, hunter and celcius).

**asmodeusBot**: A bot from a competition in Australia in 2012, the code is available [here](https://github.com/braathwaate/strategoevaluator/blob/master/agents/asmodeus/asmodeus.py). It was adapted to fit the pyspiel environment.

**hunterBot**: A bot from a competition in Australia in 2012, the code is available [here](https://github.com/braathwaate/strategoevaluator/blob/master/agents/hunter/hunter.py). It was adapted to fit the pyspiel environment. Also added the two square rule to its behavior.

**celciusBot**: Was not implemented, not working

**/rnad** & **rnadBot**: A try of using the rnad model that is talked about in [this article](https://www.science.org/stoken/author-tokens/ST-887/full). But the code available on the internet is a smaller one than the actual of the article, and also I couldn't train it enough, so the result were random and not relevant.

**gptBot**: Generated with the advices of chatGPT and using alpha-beta iterative deepening search. This bot was not really used more and considerated in the final GameStatsReplays. However it could be interesting to be tested against mcts with the same evaluations