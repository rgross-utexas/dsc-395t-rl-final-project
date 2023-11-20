import torch
from torch.distributions import Bernoulli, Categorical
from model import Model
from environment import Game, State

class Player:

    def __init__(self, model: Model) -> None:
        self.model = model

    def act(self, state: State) -> torch.Tensor:
        
        features = self._get_features(state)
        logits = self.model(features)
        action = self._get_action(logits)

        return action

    def _get_action(self, logits: torch.Tensor) -> torch.Tensor:
        
        dice = self._get_dice(logits)

        # treat the last 7 as logits a single multiclass classifier
        choice = self._get_choice(logits)

        return torch.cat((dice, choice))

    def _get_dice(self, logits: torch.Tensor) -> torch.Tensor:
        
        # treat the first 5 as logits for 5 separate binary classifiers, where
        # logit > 0 (i.e., p >.5) means 1, otherwise 0
        return self._get_dice_logits(logits) > 0

    def _get_choice(self, logits: torch.Tensor) -> torch.Tensor:
        
        # treat the last 7 as logits a single multiclass classifier
        return torch.argmax(self._get_choice_logits(logits)).view(1)

    def _get_choice_logits(self, logits) -> torch.Tensor:
        return logits[5:]

    def _get_dice_logits(self, logits) -> torch.Tensor:
        return logits[:5]

    def _get_features(self, state: State) -> torch.Tensor:

        dice = state.dice
        roll_num = torch.Tensor([state.roll_num]).int()
        chosen = state.chosen

        return torch.cat((dice, roll_num, chosen))
    
class StochasticPlayer(Player):

    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def _get_dice(self, logits: torch.Tensor) -> torch.Tensor:
        
        # treat the first 5 as logits for 5 separate binary classifiers
        return Bernoulli(self._get_dice_logits(logits)).sample()

    def _get_choice(self, logits: torch.Tensor) -> torch.Tensor:
        
        # treat the last 7 as logits a single multiclass classifier
        return Categorical(self._get_choice_logits(logits)).sample().view(1)

if __name__ == "__main__":

    # 38 output states: 1-32 is for dice selection, 33-38 is for choosing
    model = Model(num_inputs=12, num_outputs=38)
    player = Player(model=model)
    game = Game()
    print(f'{game=}')

    while not game.complete:

        state = game.get_state()
        print(f'{state=}')

        action = player.act(state=state)
        print(f'{action=}')

        keep = action[:5]
        score = action[5]

        if score.int() > 0:
            game.choose(score)