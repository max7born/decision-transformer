import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        #print(states)
        #print(actions)
        #input()
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        #print(rtg[:,:-1])

        #print(rtg[:,0])

        act_dim = action_preds.shape[2]         
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        #print('pred', action_preds)
        #print(action_target)
        #input()

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()                         # tensor
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .10)
        self.optimizer.step()
        # i=0
        # for p in self.model.parameters():
        #     print(p)
        #     i+=1
        # print(i)
        # input()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
