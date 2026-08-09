# PG-209 parameterized trace training

device=cuda; train=25; holdout=58; route holdout=['/vul/sqli/sqli_x.php', '/vul/xss/xss_04.php', '/vul/xss/xss_dom_x.php']
large body=19272990; holdout={'count': 58, 'action_accuracy': 0.63793105, 'candidate_recall': 0.44736842, 'abstain_recall': 1.0, 'unsafe_allow_count': 0, 'encoding_accuracy': 0.79310346, 'failure_accuracy': 0.58620691}; PG-208={'count': 16, 'action_accuracy': 1.0, 'candidate_recall': 1.0, 'abstain_recall': 1.0, 'unsafe_allow_count': 0, 'encoding_accuracy': 1.0, 'failure_accuracy': 1.0}
xxl body=101487169; holdout={'count': 58, 'action_accuracy': 0.89655173, 'candidate_recall': 0.84210526, 'abstain_recall': 1.0, 'unsafe_allow_count': 0, 'encoding_accuracy': 1.0, 'failure_accuracy': 0.89655173}; PG-208={'count': 16, 'action_accuracy': 1.0, 'candidate_recall': 1.0, 'abstain_recall': 1.0, 'unsafe_allow_count': 0, 'encoding_accuracy': 1.0, 'failure_accuracy': 1.0}
capacity_101m_better=True; selected=xxl

This is a route/seed holdout diagnostic. Checkpoints remain quarantined until an independent typed SQL oracle and fresh OOD gate are added.
