import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import seaborn as sns
import pandas as pd
import os
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--env',
    default='cheetah'
    )

args = parser.parse_args()
#sns.set_style("darkgrid")
mpl.style.use('seaborn')

seeds = list(range(125,129))
#seeds = [125,126,127]

fig, ax = plt.subplots()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(24)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)

alg_list = ["sac_aq", "agtr_q", "agtr_aq", "agtr_aq_2Q"]

for alg in alg_list:
	rewards = []
	for seed in seeds:
		file = os.path.join("save", args.env, str(seed), alg, 'eval.log')
		print(file)
		data = pd.read_json(file, lines=True)
		rewards.append(data['episode_reward'].to_numpy())
		timesteps = data['step'].to_numpy()

	rw_lists = np.array(rewards)
	mean_list = np.mean(rw_lists,axis = 0)
	std_list = np.std(rw_lists,axis = 0)

	ax.plot(timesteps, mean_list, label=alg)
	plt.fill_between(timesteps, mean_list + std_list,mean_list - std_list, alpha=0.2)

ax.set_xlabel("number of timesteps")
ax.set_ylabel("rewards")

plt.legend(fontsize=16, loc='center right')
plt.title(args.env,fontsize=24)

plt.savefig("{}.png".format(args.env),bbox_inches='tight')
