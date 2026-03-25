import time
try:
    import wandb
except ImportError:
    wandb = None
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
try:
    import imageio
except ImportError:
    imageio = None

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def _default_available_actions(self, n_threads):
        available_actions = []
        for agent_id in range(self.num_agents):
            if self.buffer[agent_id].available_actions is None:
                available_actions.append(None)
                continue
            act_dim = self.buffer[agent_id].available_actions.shape[-1]
            available_actions.append(
                np.ones((n_threads, act_dim), dtype=np.float32)
            )
        return available_actions

    def _extract_available_actions_from_infos(self, infos, n_threads):
        available_actions = self._default_available_actions(n_threads)
        for env_idx in range(min(len(infos), n_threads)):
            env_info = infos[env_idx]
            for agent_id in range(self.num_agents):
                if available_actions[agent_id] is None:
                    continue

                if agent_id >= len(env_info):
                    continue

                agent_info = env_info[agent_id]
                if "available_actions" not in agent_info:
                    continue

                candidate = np.asarray(
                    agent_info["available_actions"], dtype=np.float32
                ).reshape(-1)
                act_dim = available_actions[agent_id].shape[-1]
                if candidate.shape[0] != act_dim:
                    continue

                available_actions[agent_id][env_idx] = candidate

        return available_actions

    def _extract_active_masks_from_infos(self, infos, n_threads):
        active_masks = np.ones((n_threads, self.num_agents, 1), dtype=np.float32)
        for env_idx in range(min(len(infos), n_threads)):
            env_info = infos[env_idx]
            if len(env_info) == 0:
                continue
            manager_info = env_info[0]
            if "manager_active" in manager_info:
                active_masks[env_idx, 0, 0] = float(manager_info["manager_active"])
        return active_masks
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            env_infos = {
                "period_inv_cost": [],
                "period_backlog_cost": [],
                "period_inv_qty": [],
                "period_backlog_qty": [],
                "period_manager_horizon_mean": [],
                "period_queue_avg": [],
                "period_prod_cost": [],
                "period_prod_cost_per_line": [],
                "period_setup_cost": [],
                "period_pm_cost": [],
                "period_cm_cost": [],
                "period_utilization": [],
                "period_backlog_per_product": [],
                "period_inventory_per_product": [],
            }
            if self.use_linear_lr_decay:
                decayed = set()
                for agent_id in range(self.num_agents):
                    policy_obj = self.trainer[agent_id].policy
                    policy_id = id(policy_obj)
                    if policy_id in decayed:
                        continue
                    policy_obj.lr_decay(episode, episodes)
                    decayed.add(policy_id)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                available_actions = self._extract_available_actions_from_infos(
                    infos, self.n_rollout_threads
                )
                active_masks = self._extract_active_masks_from_infos(
                    infos, self.n_rollout_threads
                )
                dones_env = np.all(dones, axis=1)
                for agent_id in range(self.num_agents):
                    if available_actions[agent_id] is None:
                        continue
                    available_actions[agent_id][dones_env == True] = 1.0
                active_masks[dones_env == True] = 1.0

                for env_idx in range(min(len(infos), self.n_rollout_threads)):
                    env_info = infos[env_idx]
                    if len(env_info) == 0:
                        continue
                    agent0_info = env_info[0]
                    if "period_inv_cost" in agent0_info:
                        env_infos["period_inv_cost"].append(
                            float(agent0_info["period_inv_cost"])
                        )
                    if "period_backlog_cost" in agent0_info:
                        env_infos["period_backlog_cost"].append(
                            float(agent0_info["period_backlog_cost"])
                        )
                    if "period_manager_horizon_mean" in agent0_info:
                        env_infos["period_manager_horizon_mean"].append(
                            float(agent0_info["period_manager_horizon_mean"])
                        )
                    if "period_manager_horizons" in agent0_info:
                        per_line = np.asarray(
                            agent0_info["period_manager_horizons"], dtype=np.float32
                        )
                        for line_idx, val in enumerate(per_line):
                            key = f"horizon_line_{line_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                    if "period_inv_qty" in agent0_info:
                        env_infos["period_inv_qty"].append(
                            float(agent0_info["period_inv_qty"])
                        )
                    if "period_backlog_qty" in agent0_info:
                        env_infos["period_backlog_qty"].append(
                            float(agent0_info["period_backlog_qty"])
                        )
                    if "period_backlog_per_product" in agent0_info:
                        per_prod = np.asarray(
                            agent0_info["period_backlog_per_product"], dtype=np.float32
                        )
                        for prod_idx, val in enumerate(per_prod):
                            key = f"backlog_prod_{prod_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                        env_infos["period_backlog_per_product"].append(
                            float(np.mean(per_prod))
                        )
                    if "period_inventory_per_product" in agent0_info:
                        per_prod = np.asarray(
                            agent0_info["period_inventory_per_product"], dtype=np.float32
                        )
                        for prod_idx, val in enumerate(per_prod):
                            key = f"inventory_prod_{prod_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                        env_infos["period_inventory_per_product"].append(
                            float(np.mean(per_prod))
                        )
                    if "period_prod_cost" in agent0_info:
                        env_infos["period_prod_cost"].append(
                            float(agent0_info["period_prod_cost"])
                        )
                    if "period_prod_cost_per_line" in agent0_info:
                        per_line = np.asarray(
                            agent0_info["period_prod_cost_per_line"], dtype=np.float32
                        )
                        for line_idx, val in enumerate(per_line):
                            key = f"prod_cost_line_{line_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                        env_infos["period_prod_cost_per_line"].append(
                            float(np.mean(per_line))
                        )
                    if "period_setup_cost" in agent0_info:
                        env_infos["period_setup_cost"].append(
                            float(agent0_info["period_setup_cost"])
                        )
                    if "period_pm_cost" in agent0_info:
                        env_infos["period_pm_cost"].append(
                            float(agent0_info["period_pm_cost"])
                        )
                    if "period_cm_cost" in agent0_info:
                        env_infos["period_cm_cost"].append(
                            float(agent0_info["period_cm_cost"])
                        )
                    if "period_utilization" in agent0_info:
                        env_infos["period_utilization"].append(
                            float(agent0_info["period_utilization"])
                        )
                    if "period_utilization_per_line" in agent0_info:
                        per_line = np.asarray(
                            agent0_info["period_utilization_per_line"], dtype=np.float32
                        )
                        for line_idx, val in enumerate(per_line):
                            key = f"util_line_{line_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                    if "period_queue_avg_per_line" in agent0_info:
                        per_line = np.asarray(
                            agent0_info["period_queue_avg_per_line"], dtype=np.float32
                        )
                        for line_idx, val in enumerate(per_line):
                            key = f"queue_avg_line_{line_idx}"
                            env_infos.setdefault(key, []).append(float(val))
                        env_infos["period_queue_avg"].append(
                            float(np.mean(per_line))
                        )

                data = obs, rewards, dones, infos, available_actions, active_masks, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # Episode reward (mean across threads)
            total_reward_per_env = np.zeros(self.n_rollout_threads, dtype=np.float32)
            for agent_id in range(self.num_agents):
                ep_rewards = np.sum(
                    self.buffer[agent_id].rewards, axis=0
                ).reshape(-1)
                train_infos[agent_id]["episode_reward"] = float(
                    np.mean(ep_rewards)
                )
                total_reward_per_env += ep_rewards
            env_infos["episode_reward_total"] = [float(np.mean(total_reward_per_env))]
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                scenario_name = getattr(
                    self.all_args,
                    "scenario_name",
                    getattr(self.all_args, "env_name", "BOSCH"),
                )
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for count, env_info in enumerate(infos):
                            if 'individual_reward' in env_info[agent_id].keys():
                                idv_rews.append(env_info[agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        available_actions = self._default_available_actions(self.n_rollout_threads)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            if available_actions[agent_id] is not None:
                self.buffer[agent_id].available_actions[0] = available_actions[
                    agent_id
                ].copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            agent_available_actions = None
            if self.buffer[agent_id].available_actions is not None:
                agent_available_actions = self.buffer[agent_id].available_actions[step]
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            agent_available_actions)
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        if getattr(self.all_args, "debug_actions", False):
            max_steps = int(getattr(self.all_args, "debug_action_steps", 0))
            if step < max_steps and self.n_rollout_threads > 0:
                mgr = actions[0][0]
                mgr_products = [int(x) for x in mgr[0::2]]
                mgr_horizons = [int(x) for x in mgr[1::2]]
                print(
                    f"[debug] step {step} manager products={mgr_products} horizons={mgr_horizons}"
                )
                machine_acts = [
                    int(actions[agent_id][0][0])
                    for agent_id in range(1, self.num_agents)
                ]
                print(
                    f"[debug] step {step} machine actions={machine_acts}"
                )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, active_masks, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        for agent_id in range(self.num_agents):
            done_agent = dones[:, agent_id] == True
            if np.any(done_agent):
                rnn_states[agent_id][done_agent] = np.zeros(
                    (done_agent.sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                rnn_states_critic[agent_id][done_agent] = np.zeros(
                    (done_agent.sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[agent_id],
                                        rnn_states_critic[agent_id],
                                        actions[agent_id],
                                        action_log_probs[agent_id],
                                        values[agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id],
                                        active_masks=active_masks[:, agent_id],
                                        available_actions=available_actions[agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        eval_available_actions = self._default_available_actions(
            self.n_eval_rollout_threads
        )

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                agent_available_actions = eval_available_actions[agent_id]
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                agent_available_actions,
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_available_actions = self._extract_available_actions_from_infos(
                eval_infos, self.n_eval_rollout_threads
            )
            eval_dones_env = np.all(eval_dones, axis=1)
            for agent_id in range(self.num_agents):
                if eval_available_actions[agent_id] is None:
                    continue
                eval_available_actions[agent_id][eval_dones_env == True] = 1.0
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        if self.all_args.save_gifs and imageio is None:
            raise RuntimeError(
                "save_gifs=True requires imageio. Install it with `pip install imageio`."
            )
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            render_available_actions = self._default_available_actions(
                self.n_rollout_threads
            )
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    agent_available_actions = render_available_actions[agent_id]
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        agent_available_actions,
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                render_available_actions = self._extract_available_actions_from_infos(
                    infos, self.n_rollout_threads
                )
                dones_env = np.all(dones, axis=1)
                for agent_id in range(self.num_agents):
                    if render_available_actions[agent_id] is None:
                        continue
                    render_available_actions[agent_id][dones_env == True] = 1.0
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
