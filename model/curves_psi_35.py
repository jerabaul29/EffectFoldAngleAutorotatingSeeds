import matplotlib.pyplot as plt
from experimental_data import KL_list, St_list, Re_list
import matplotlib
import pickle

print(KL_list)
print(St_list)
print(Re_list)

with open('model_data_psi_35.pkl', 'rb') as handle:
    dict_model_data = pickle.load(handle)

w, h = plt.figaspect(1.55)
fig, ax = plt.subplots(figsize=(w, h))
matplotlib.rcParams.update({'font.size': 12})

plt.plot(dict_model_data["KL"], -dict_model_data["Re"], '*', markersize='10', label="model $\psi = 35$", color='blue')
plt.scatter(KL_list[4:], Re_list[4:], color='red', label="experiments $\psi = 35$")
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
plt.savefig('Figures/Re_35.pdf')

plt.figure()
# plt.scatter(KL_list[4:], St_list[4:], label="experiments $\psi = 35")
# the first points are for psi = 0
nbr_untrusted_points = 4
plt.errorbar(KL_list[nbr_untrusted_points:], St_list[nbr_untrusted_points:], yerr=0.15, label="experiments $\psi = 35$", marker="*", markersize=10, linestyle="")
plt.plot(dict_model_data["KL"], dict_model_data["St"], label="model $\psi = 35$")
plt.legend()

plt.savefig('Figures/St_35.pdf')

plt.show()
