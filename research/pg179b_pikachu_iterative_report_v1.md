# PG-179B Pikachu iterative GET/POST probe

episodes: 7; steps: 35; GET/POST steps: 11/24
parameterized episodes: GET=1, POST=6; dual-channel=6; invented parameter names=false
adaptive branch: {'probe_candidate_other_method': 0, 'repeat_matched_negative_pair': 6, 'abstain_unknown_oracle': 1}; candidate signals: 2; typed positives: 0

所有输入是字母数字 canary；字段只来自浏览器观察到的 request schema。回显/SQL-looking/跳转仅是 candidate signal，未获得 typed oracle，因此全部 abstain，禁止训练和长期记忆。
