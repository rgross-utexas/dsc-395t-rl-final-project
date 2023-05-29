import logging

import torch

class Game:

    class State:
        
        def __init__(self, scores, dice, roll_num) -> None:
            self.chosen = scores > -1
            self.dice = dice
            self.roll_num = roll_num

        def __repr__(self) -> str:
            return f'{self.roll_num=}, {self.dice=}, {self.chosen=}'

    MAX_ROLLS = 3
    NUM_DICE = 5
    NUM_SIDES = 6
    CHEAT_SCORE = -10
    ZEROS = torch.zeros(NUM_DICE).int()

    def __init__(self) -> None:
        
        self.scores = torch.zeros(6).int() - 1
        self.dice = Game.ZEROS
        self.roll_num = 0
        self.cheat = False
        self.complete = False

    def __repr__(self) -> str:
        return f'{self.scores=}, {self.dice=}, {self.roll_num=}, {self.cheat}, {self.complete}, {self.get_total_score()}'

    def roll(self, keep: torch.Tensor = ZEROS.bool()) -> None:

        if self.complete:
            raise Exception(f'Unable to roll because game is complete! {self.game=}')

        if not self._can_roll():

            # logging.error(f'Unable to roll! Marking game as complete due to illegal action!')

            self.cheat = True
            self.complete = True

            return torch.tensor(Game.CHEAT_SCORE)
        
        roll = torch.randint(low=1, high=Game.NUM_SIDES, size=(Game.NUM_DICE,))
        self.dice = roll*torch.logical_not(keep) + self.dice*keep

        self.roll_num += 1

        return torch.tensor(0)

    def choose(self, choice: int) -> torch.Tensor:

        if self.complete:
            return

        if not self._can_choose(choice):
            self.cheat = True
            self.complete = True

            return torch.tensor(Game.CHEAT_SCORE)

        score = (self.dice * (self.dice == (choice))).sum()
        self.scores[choice-1] = score

        if torch.all(self.scores > -1):
            # all the scoring has been completed, so mark game complete
            self.complete = True
        else:
            # restart turn
            self.roll_num = 0
            self.dice = Game.ZEROS
        
        return score

    def apply_action(self, action: torch.Tensor) -> int:

        if self.is_action_roll(action):

            # if action is 0-31, this is choosing which dice to keep
            keep = []
            for i in range(Game.NUM_DICE):
                keep.append((int(action) >> i) % 2)

            keep = torch.Tensor(keep).bool()
            # print(f'{keep=}')
            return self.roll(keep)
        else:
            # if action is 32-37, this is choosing which dice to score
            action = int(action) - 2**Game.NUM_DICE + 1
            return self.choose(action)

    def is_new_turn(self) -> bool:
        return self.roll_num == 0

    def is_action_roll(self, action) -> bool:
        return action < 2**Game.NUM_DICE

    def is_action_choice(self, action) -> bool:
        return not self.is_action_roll(action)

    def get_total_score(self) -> int:

        if self.cheat:
            return Game.CHEAT_SCORE
        else:
            return (self.scores*(self.scores != -1)).sum()

    def get_reward(self) -> int:
        return self.get_total_score()
    
    def get_state(self) -> State:
        return Game.State(self.scores, self.dice, self.roll_num)

    def _can_roll(self) -> bool:
        return self.roll_num <= Game.MAX_ROLLS

    def _can_choose(self, choice: int) -> bool:
        return self.scores[choice-1] == -1


if __name__ == "__main__":

    state = Game()

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(1)
    print(f'Turn 1: {state}')

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(2)
    print(f'Turn 2: {state}')

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(3)
    print(f'Turn 3: {state}')

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(4)
    print(f'Turn 4: {state}')

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(5)
    print(f'Turn 5: {state}')

    state.roll()
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.roll(keep=torch.Tensor([1, 0 , 0, 0, 0]).int())
    print(state)
    state.choose(6)
    print(f'Turn 6: {state}')
