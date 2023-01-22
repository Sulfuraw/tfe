# Slovenian Tarok Card Game
[Slovenian Tarok](https://en.wikipedia.org/wiki/K%C3%B6nigrufen#Slovenia) is a variant of central European [Tarot card games](https://en.wikipedia.org/wiki/Tarot_card_games). It is essentially a three- or four-player, trick-taking, competitive game of skill with bidding. Computationally speaking, Tarok is moderately more complex than Bridge [[1]](#references). Detailed game rules are available at https://www.pagat.com/tarot/sltarok.html.

The environment was implemented by [Nejc Ilenic](https://github.com/inejc) and [Tim Smole](https://github.com/TimSmole); the original repository is available at https://github.com/semanticweights/tarok.

### Implementation Notes
Note that the current implementation is a full game without the [announcements](https://www.pagat.com/tarot/sltarok.html#announcements). The game is fully playable nevertheless as announcements can be considered an optional addition which will be added in a future PR.

Furthermore, the environment is implemented in an implicitly stochastic manner, i.e. chance node (for dealing the cards) returns a single dummy action and applying it utilizes an internal RNG to deal all of the cards at once (within that single action). The reasoning for that particular game is that implicit implementation seemed easier (mostly meaning that less code had to be written). In addition, any algorithm that relies on the explicit game tree likely isn't viable for a game this large.

### References
- [1] [Luštrek Mitja, Matjaž Gams, Ivan Bratko. "A program for playing Tarok." ICGA journal 26.3 (2003): 190-197.](https://pdfs.semanticscholar.org/a920/70fe11f75f58c27ed907c4688747259cae15.pdf)
