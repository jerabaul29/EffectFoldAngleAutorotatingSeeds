import matplotlib.pyplot as plt
from experimental_data import total_Re, total_strouhal, KL_total
from experimental_data import KL_list, St_list, Re_list
import matplotlib
import pickle

###########################################################
# load experimental data
with open('model_data_psi_35.pkl', 'rb') as handle:
    dict_model_data_35 = pickle.load(handle)

with open('model_data_psi_0.pkl', 'rb') as handle:
    dict_model_data_0 = pickle.load(handle)

###########################################################
# figure with the Re for all kinds of experiments
w, h = plt.figaspect(1.00)
fig, ax = plt.subplots(figsize=(w, h))
matplotlib.rcParams.update({'font.size': 12})

plt.plot(dict_model_data_35["KL"], -dict_model_data_35["Re"], markersize='10', label="model $\psi = 35$", color='blue', linestyle="-")
plt.plot(KL_list[4:], Re_list[4:], marker='*', markersize=10, color='red', label="experiments $\psi = 35$", linestyle="")

plt.plot(dict_model_data_0["KL"], -dict_model_data_0["Re"], markersize='10', label="model $\psi = 0$", color='black', linestyle="--")
plt.plot(KL_total, -total_Re, marker='o', markersize='10', label="experiments $\psi = 0$", color='orange', linestyle="")

# plt.grid()
plt.xlim([-0.05, 4.7])
plt.ylim([-14000, -2000])
plt.xlabel('KL')
plt.ylabel('Re')
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
plt.tight_layout()
# plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
plt.savefig('Figures/Re_both.pdf')

############################################
# figure for the St for all experiments
plt.figure()
# plt.scatter(KL_list[4:], St_list[4:], label="experiments $\psi = 35")
# the first points are for psi = 0
nbr_untrusted_points = 4
plt.plot(dict_model_data_35["KL"], dict_model_data_35["St"], label="model $\psi = 35$", color='blue', linestyle="-")
plt.plot([], [], label="experiments $\psi = 35$", marker="*", markersize=10, linestyle="", color='red')
plt.errorbar(KL_list[nbr_untrusted_points:], St_list[nbr_untrusted_points:], yerr=0.15, marker="*", markersize=10, linestyle="", color='red')

nbr_untrusted_points = 0
plt.plot(dict_model_data_0["KL"], dict_model_data_0["St"], label="model $\psi = 0$", color="black", linestyle="--")
plt.errorbar(KL_total[nbr_untrusted_points:], total_strouhal[nbr_untrusted_points:], yerr=0.15, label="experiments $\psi = 0$", marker="o", markersize=10, linestyle="", color='orange')

plt.xlabel('KL')
plt.ylabel('St')

plt.ylim([0.1, 1.4])
plt.legend(loc='upper left', frameon=False)

plt.tight_layout()

plt.savefig('Figures/St_both.pdf')

############################################
# show all
plt.show()
