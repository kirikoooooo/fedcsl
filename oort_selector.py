# -*- coding: utf-8 -*-
"""
Oort 风格客户端选择器（与 Oort OSDI'21 训练选择逻辑兼容，可无缝接入 FedCSL）。
不依赖 Oort 包，仅复用其 UCB + 探索 + Pacer 的选客户端逻辑。
"""
import math
import numpy as np
from random import Random
from collections import OrderedDict


def _make_oort_args(
    exploration_factor=0.9,
    exploration_decay=0.95,
    exploration_min=0.2,
    exploration_alpha=0.3,
    round_threshold=10.0,
    round_penalty=2.0,
    pacer_step=20,
    pacer_delta=5.0,
    sample_window=5.0,
    clip_bound=0.98,
    cut_off_util=0.7,
    blacklist_rounds=-1,
    blacklist_max_len=0.3,
    **kwargs
):
    """构造 Oort 选择器所需的 args 命名空间。"""
    class Args:
        pass
    args = Args()
    args.exploration_factor = exploration_factor
    args.exploration_decay = exploration_decay
    args.exploration_min = exploration_min
    args.exploration_alpha = exploration_alpha
    args.round_threshold = round_threshold
    args.round_penalty = round_penalty
    args.pacer_step = pacer_step
    args.pacer_delta = pacer_delta
    args.sample_window = sample_window
    args.clip_bound = clip_bound
    args.cut_off_util = cut_off_util
    args.blacklist_rounds = blacklist_rounds
    args.blacklist_max_len = blacklist_max_len
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


class OortTrainingSelector:
    """
    Oort 训练侧客户端选择器接口：
    - register_client(client_id, feedbacks): 注册客户端，feedbacks 含 reward, duration
    - update_client_util(client_id, feedbacks): 每轮结束后更新 reward/duration
    - select_participant(num_of_clients, feasible_clients): 返回本轮被选中的客户端 id 列表
    """

    def __init__(self, args=None, sample_seed=233):
        if args is None:
            args = _make_oort_args()
        self.totalArms = OrderedDict()
        self.training_round = 0
        self.exploration = getattr(args, 'exploration_factor', 0.9)
        self.decay_factor = getattr(args, 'exploration_decay', 0.95)
        self.exploration_min = getattr(args, 'exploration_min', 0.2)
        self.alpha = getattr(args, 'exploration_alpha', 0.3)
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.args = args
        self.round_threshold = getattr(args, 'round_threshold', 10.0)
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0
        self.sample_window = getattr(args, 'sample_window', 5.0)
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None
        np.random.seed(sample_seed)

    def register_client(self, clientId, feedbacks):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {
                'reward': feedbacks['reward'],
                'duration': feedbacks['duration'],
                'time_stamp': self.training_round,
                'count': 0,
                'status': True,
            }
            self.unexplored.add(clientId)

    def _calculate_sum_util(self, clientList):
        cnt, cntUtil = 1e-4, 0.0
        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']
        return cntUtil / cnt

    def _pacer(self):
        lastExplorationUtil = self._calculate_sum_util(self.exploreClients)
        lastExploitationUtil = self._calculate_sum_util(self.exploitClients)
        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)
        self.successfulClients = set()
        pacer_step = getattr(self.args, 'pacer_step', 20)
        pacer_delta = getattr(self.args, 'pacer_delta', 5.0)
        if self.training_round >= 2 * pacer_step and self.training_round % pacer_step == 0:
            utilLast = sum(self.exploitUtilHistory[-2 * pacer_step:-pacer_step])
            utilCurrent = sum(self.exploitUtilHistory[-pacer_step:])
            if abs(utilCurrent - utilLast) <= utilLast * 0.1:
                self.round_threshold = min(100.0, self.round_threshold + pacer_delta)
                self.last_util_record = self.training_round - pacer_step
            elif len(self.exploitUtilHistory) >= 2 and abs(utilCurrent - utilLast) >= utilLast * 0.5:
                self.round_threshold = max(pacer_delta, self.round_threshold - pacer_delta)
                self.last_util_record = self.training_round - pacer_step

    def update_client_util(self, clientId, feedbacks):
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = feedbacks.get('status', True)
        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)

    def _get_blacklist(self):
        blacklist = []
        blacklist_rounds = getattr(self.args, 'blacklist_rounds', -1)
        blacklist_max_len = getattr(self.args, 'blacklist_max_len', 0.3)
        if blacklist_rounds != -1:
            sorted_ids = sorted(
                list(self.totalArms.keys()),
                reverse=True,
                key=lambda k: self.totalArms[k]['count']
            )
            for clientId in sorted_ids:
                if self.totalArms[clientId]['count'] > blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break
            max_len = blacklist_max_len * len(self.totalArms)
            if len(blacklist) > max_len:
                blacklist = blacklist[:int(max_len)]
        return set(blacklist)

    def _get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList = sorted(aList)
        clip_value = aList[min(int(len(aList) * clip_bound), len(aList) - 1)] if aList else 0.0
        _max = max(aList) if aList else 1.0
        _min = min(aList) * 0.999 if aList else 0.0
        _range = max(_max - _min, thres)
        _avg = sum(aList) / max(1e-4, float(len(aList))) if aList else 0.0
        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)

    def select_participant(self, num_of_clients, feasible_clients=None):
        viable = feasible_clients if feasible_clients is not None else set(
            k for k, v in self.totalArms.items() if v['status']
        )
        return self._get_top_k(num_of_clients, self.training_round + 1, viable)

    def _get_top_k(self, numOfSamples, cur_time, feasible_clients):
        self.training_round = cur_time
        self.blacklist = self._get_blacklist()
        self._pacer()
        round_penalty = getattr(self.args, 'round_penalty', 2.0)
        cut_off_util = getattr(self.args, 'cut_off_util', 0.7)
        clip_bound = getattr(self.args, 'clip_bound', 0.98)

        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if x in feasible_clients and x not in self.blacklist]
        if not orderedKeys:
            return list(feasible_clients)[:numOfSamples] if feasible_clients else []

        if self.round_threshold < 100.0:
            sorted_duration = sorted([self.totalArms[k]['duration'] for k in client_list])
            idx = min(int(len(sorted_duration) * self.round_threshold / 100.0), len(sorted_duration) - 1)
            self.round_prefer_duration = sorted_duration[idx]
        else:
            self.round_prefer_duration = float('inf')

        moving_reward, staleness = [], []
        for clientId in orderedKeys:
            if self.totalArms[clientId]['reward'] > 0:
                moving_reward.append(self.totalArms[clientId]['reward'])
                staleness.append(cur_time - self.totalArms[clientId]['time_stamp'])

        if not moving_reward:
            picked = list(orderedKeys)[:numOfSamples]
            while len(picked) < numOfSamples and len(picked) < len(orderedKeys):
                c = self.rng.choice(orderedKeys)
                if c not in picked:
                    picked.append(c)
            return picked[:numOfSamples]

        max_r, min_r, range_r, avg_r, clip_value = self._get_norm(moving_reward, clip_bound)
        max_s, min_s, range_s, avg_s, _ = self._get_norm(staleness, thres=1)

        scores = {}
        for key in orderedKeys:
            if self.totalArms[key]['count'] > 0:
                creward = min(self.totalArms[key]['reward'], clip_value)
                sc = (creward - min_r) / float(range_r) + math.sqrt(
                    0.1 * math.log(max(cur_time, 1)) / max(1, self.totalArms[key]['time_stamp'])
                )
                dur = self.totalArms[key]['duration']
                if dur > self.round_prefer_duration and self.round_prefer_duration < 1e10:
                    sc *= (float(self.round_prefer_duration) / max(1e-4, dur)) ** round_penalty
                scores[key] = sc
            else:
                scores[key] = 0.0

        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        exploitLen = min(int(numOfSamples * (1.0 - self.exploration)), len(clientLakes))
        exploitLen = max(1, exploitLen)

        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)
        cut_off = scores[sortedClientUtil[exploitLen - 1]] * cut_off_util
        pickedClients = [c for c in sortedClientUtil if scores[c] >= cut_off]

        totalSc = max(1e-4, sum(scores[k] for k in pickedClients))
        probs = [scores[k] / totalSc for k in pickedClients]
        n_pick = min(exploitLen, len(pickedClients))
        if n_pick >= len(pickedClients):
            self.exploitClients = list(pickedClients)
        else:
            self.exploitClients = list(np.random.choice(pickedClients, size=n_pick, replace=False, p=probs))

        if self.unexplored:
            _unexplored = [x for x in self.unexplored if x in feasible_clients]
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                dur = self.totalArms[cl]['duration']
                if dur > self.round_prefer_duration and self.round_prefer_duration < 1e10:
                    init_reward[cl] *= (float(self.round_prefer_duration) / max(1e-4, dur)) ** round_penalty
            exploreLen = min(len(_unexplored), numOfSamples - len(self.exploitClients))
            exploreLen = min(exploreLen, int(self.sample_window * exploreLen))
            if _unexplored and exploreLen > 0:
                sorted_unex = sorted(init_reward, key=init_reward.get, reverse=True)
                pickedUnexplored = sorted_unex[:min(exploreLen, len(sorted_unex))]
                unexploredSc = sum(init_reward[k] for k in pickedUnexplored)
                if unexploredSc > 1e-12:
                    p_unex = [init_reward[k] / unexploredSc for k in pickedUnexplored]
                    n_ux = min(exploreLen, len(pickedUnexplored))
                    extra = list(np.random.choice(pickedUnexplored, size=n_ux, replace=False, p=p_unex))
                    self.exploreClients = extra
                    pickedClients = self.exploitClients + extra
                else:
                    pickedClients = self.exploitClients + pickedUnexplored[:exploreLen]
            else:
                pickedClients = list(self.exploitClients)
        else:
            self.exploration_min = 0.0
            self.exploration = 0.0
            pickedClients = list(self.exploitClients)

        while len(pickedClients) < numOfSamples and len(pickedClients) < len(orderedKeys):
            c = self.rng.choice(orderedKeys)
            if c not in pickedClients:
                pickedClients.append(c)

        return list(pickedClients)[:numOfSamples]


def create_oort_selector(config=None, sample_seed=233):
    """从 config 或默认值创建 Oort 选择器，便于在 FedCSL_All 中无缝使用。"""
    cfg = config or {}
    oort_cfg = cfg.get('oort', {})
    args = _make_oort_args(
        exploration_factor=oort_cfg.get('exploration_factor', 0.9),
        exploration_decay=oort_cfg.get('exploration_decay', 0.95),
        exploration_min=oort_cfg.get('exploration_min', 0.2),
        exploration_alpha=oort_cfg.get('exploration_alpha', 0.3),
        round_threshold=oort_cfg.get('round_threshold', 10.0),
        round_penalty=oort_cfg.get('round_penalty', 2.0),
        pacer_step=oort_cfg.get('pacer_step', 20),
        pacer_delta=oort_cfg.get('pacer_delta', 5.0),
        sample_window=oort_cfg.get('sample_window', 5.0),
        clip_bound=oort_cfg.get('clip_bound', 0.98),
        cut_off_util=oort_cfg.get('cut_off_util', 0.7),
        blacklist_rounds=oort_cfg.get('blacklist_rounds', -1),
        blacklist_max_len=oort_cfg.get('blacklist_max_len', 0.3),
    )
    return OortTrainingSelector(args=args, sample_seed=sample_seed)
