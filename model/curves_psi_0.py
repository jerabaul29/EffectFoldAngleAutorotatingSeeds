import numpy as np
import matplotlib.pyplot as plt
from experimental_data import total_Re, total_strouhal, KL_total
import matplotlib
import pickle

with open('model_data_psi_0.pkl', 'rb') as handle:
    dict_model_data = pickle.load(handle)

w, h = plt.figaspect(1.55)
fig, ax = plt.subplots(figsize=(w, h))
matplotlib.rcParams.update({'font.size': 12})

plt.plot(dict_model_data["KL"], -dict_model_data["Re"], '*', markersize='10', label="model $\psi = 0$", color='blue')
plt.plot(KL_total, -total_Re, '*', markersize='10', label="experiments $\psi = 0$", color='red')
# plt.grid()
plt.ylim([-14000, 0])
plt.xlabel('KL')
#
ax.yaxis.set_label_position('right')
ax.yaxis.set_ticks_position('right')
ax.tick_params(labelleft='off', labelright='on')
#
# ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
#
plt.legend(loc='lower right', frameon=False)
#
plt.xlim([0.2, 4.3])
# plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
plt.savefig('Figures/Re_0.pdf')

plt.figure()
# do not trust the first points for the St value
nbr_untrusted_points = 5
plt.errorbar(KL_total[nbr_untrusted_points:], total_strouhal[nbr_untrusted_points:], yerr=0.15, label="experiments $\psi = 0$", marker="*", markersize=10, linestyle="")
plt.plot(dict_model_data["KL"], dict_model_data["St"], label="model $\psi = 0$")
plt.ylim([0, 0.9])
plt.legend()

plt.savefig('Figures/St_0.pdf')

plt.show()
