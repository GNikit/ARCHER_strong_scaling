# %%
from extract_lb_data import ExtractLB
import os
from math import ceil
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tikzplotlib import save as tikz_save
from fluidity_tools import stat_parser as stat
import numpy as np

# Make background white
plt.style.use('default')

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
matplotlib.rc('font', **{'family': 'sans-serif',
                         'sans-serif': ['DejaVu Sans'], 'size': 10})

# Set the font used for MathJax - more on this later
matplotlib.rc('mathtext', **{'default': 'regular'})

MAX_ADAPT = 4
SAVE = False
# %% [markdown]

# Change dir to data location
# Examine if group sets are more efficient that single groupset
# This is a bit of a cheat since we are comparing load balanced group sets with themselves rather than non DLBed ones
# %%
# os.chdir('/home/gn/Code/Python/lb_analysis')
# grpset1_vars = ExtractLB()
# grpset1_vars.get_run_stats('physor/exotic/group_lb',
#                            'rad_c5g7mox2D_1grp_hwn_reg_adapt_mf_lb_no_io_full_conv  ')
# grpset3_vars = ExtractLB()
# grpset3_vars.get_run_stats('physor/exotic/group_lb',
#                            'rad_c5g7mox2D_3grp_hwn_reg_adapt_mf_lb_no_io_full_conv')
#
# print(
#     f'Runtime Improvement {ceil((grpset1_vars.WALL_T[1][-1] - grpset3_vars.WALL_T[1][-1]) / grpset1_vars.WALL_T[1][-1] * 100)}%')
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(range(1, MAX_ADAPT + 1),
#         grpset1_vars.WALL_T[1][:MAX_ADAPT], color='blue', label='1 group set', marker='o')
# ax.plot(range(1, MAX_ADAPT + 1),
#         grpset3_vars.WALL_T[1][:MAX_ADAPT], color='red', label='3 group sets', marker='s')
#
# ax.set(xlabel='Adapt step', ylabel='Wall Time [s]')
# ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
# ax.legend()
# fig.tight_layout()
#

# %% [markdown]
# ## Comparison between load balanced and non load balanced multigroup set solves
# ## with multiple number of group sets and a single number of group sets

# %%
# No DLB, convergence of flux and keff at 1e-10
root_dir = '/home/gn/Code/Archer/tests/c5g7mox2D/group_full_flux_low_adapt_tol'
grpset1_vars = ExtractLB()
stat_f = 'rad_c5g7mox2D_1grp_hwn_reg_adapt_mf_no_io'
grpset1_vars.get_run_stats(root_dir, stat_f, solver_t=True)

grpset2_vars = ExtractLB()
stat_f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_no_io'
grpset2_vars.get_run_stats(root_dir, stat_f, solver_t=True)

grpset3_vars = ExtractLB()
stat_f = 'rad_c5g7mox2D_3grp_hwn_reg_adapt_mf_no_io'
grpset3_vars.get_run_stats(root_dir, stat_f, solver_t=True)

# grpset7_vars = ExtractLB()
# stat_f = 'rad_c5g7mox2D_7grp_hwn_reg_adapt_mf_no_io'
# grpset7_vars.get_run_stats(root_dir, stat_f, solver_t=True)

# DLB on, convergence of flux and keff at 1e-10
root_dir = '/home/gn/Code/Archer/tests/c5g7mox2D/group_full_lb_flux_low_adapt_tol/'
grpset1_vars_lb = ExtractLB()
stat_f = 'rad_c5g7mox2D_1grp_hwn_reg_adapt_mf_no_io'
grpset1_vars_lb.get_run_stats(root_dir, stat_f, solver_t=True)

grpset2_vars_lb = ExtractLB()
stat_f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_no_io'
grpset2_vars_lb.get_run_stats(root_dir, stat_f, solver_t=True)

grpset3_vars_lb = ExtractLB()
stat_f = 'rad_c5g7mox2D_3grp_hwn_reg_adapt_mf_no_io'
grpset3_vars_lb.get_run_stats(root_dir, stat_f, solver_t=True)

# grpset7_vars_lb = ExtractLB()
# stat_f = 'rad_c5g7mox2D_7grp_hwn_reg_adapt_mf_no_io'
# grpset7_vars_lb.get_run_stats(root_dir, stat_f, solver_t=True)

# %% [markdown]
# ## Average DOFs vs Adapt order
# %%
# Plot DOFs vs adapt
reference = 1.186550
fig, ax = plt.subplots()
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars.ALLDOFS[1][:MAX_ADAPT]*7,
        label='1 group set', color='b', marker='o', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars.ALLDOFS[1][:MAX_ADAPT]*7/2,
        label='2 group set', color='g', marker='v', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars.ALLDOFS[1][:MAX_ADAPT]*7/3,
        label='3 group set', color='m', marker='s', markerfacecolor='none')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars.ALLDOFS[1][:MAX_ADAPT], label='7 group set', color='r', marker='x', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars_lb.ALLDOFS[1][:MAX_ADAPT]
        * 7, label='1 group set DLB', color='b', marker='o', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars_lb.ALLDOFS[1][:MAX_ADAPT]
        * 7/2, label='2 group set DLB', color='g', marker='v', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars_lb.ALLDOFS[1][:MAX_ADAPT]
        * 7/3, label='3 group set DLB', color='m', marker='s', linestyle='--')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars_lb.ALLDOFS[1][:MAX_ADAPT], label='7 group set DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Adapt step', ylabel='Av. DOFs per spatial node')
ax.set_xticks(range(1, MAX_ADAPT + 1))
ax.legend()
fig.tight_layout()
if SAVE:
    fig_name = 'c5g7_dofs_vs_adapt'
    tikz_save(f'/home/gn/Dropbox/PhD/figures/{fig_name}.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


# %% [markdown]
# ## Walltime vs adapt order
# %%
# Plot the Walltime vs the adapt steps
fig, ax = plt.subplots()
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars.WALL_T[1][:MAX_ADAPT],
        label='1 group set', color='b', marker='o', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars.WALL_T[1][:MAX_ADAPT],
        label='2 group set', color='g', marker='v', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars.WALL_T[1][:MAX_ADAPT],
        label='3 group set', color='m', marker='s', markerfacecolor='none')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars.WALL_T[1][:MAX_ADAPT],
# label='7 group set', color='r', marker='x', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars_lb.WALL_T[1][:MAX_ADAPT],
        label='1 group set DLB', color='b', marker='o', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars_lb.WALL_T[1][:MAX_ADAPT],
        label='2 group set DLB', color='g', marker='v', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars_lb.WALL_T[1][:MAX_ADAPT],
        label='3 group set DLB', color='m', marker='s', linestyle='--')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars_lb.WALL_T[1][:MAX_ADAPT],
# label='7 group set DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Adapt step', ylabel='Wall Time [s]')
ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()
if SAVE:
    fig_name = 'c5g7_walltime_vs_adapt'
    tikz_save(f'/home/gn/Dropbox/PhD/figures/{fig_name}.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')

# %% [markdown]
# ## Load balance time vs Adapt order
# %%
# Plot the Load balance time vs adapt steps
fig, ax = plt.subplots()
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars_lb.LB_T[1][:MAX_ADAPT],
        label='1 group DLB', color='b', marker='o', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars_lb.LB_T[1][:MAX_ADAPT],
        label='2 group DLB', color='g', marker='v', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars_lb.LB_T[1][:MAX_ADAPT],
        label='3 group DLB', color='m', marker='s', linestyle='--')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars_lb.LB_T[1][:MAX_ADAPT],
# label='7 group DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Adapt step', ylabel='DLB Time [s]')
ax.set_xticks(range(1, MAX_ADAPT + 1))
ax.set_yscale('log')
ax.legend()
fig.tight_layout()


# %% [markdown]
# ## Solver time vs Adapt order
# %%
# Plot the Solver time vs adapt steps
fig, ax = plt.subplots()

# Non-load balancing solver times
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars.SOLVER_T[1][:MAX_ADAPT],
        label='1 group set', color='b', marker='o', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars.SOLVER_T[1][:MAX_ADAPT],
        label='2 group set', color='g', marker='v', markerfacecolor='none')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars.SOLVER_T[1][:MAX_ADAPT],
        label='3 group set', color='m', marker='s', markerfacecolor='none')

# Load balancing solver times
ax.plot(range(1, MAX_ADAPT+1), grpset1_vars_lb.SOLVER_T[1][:MAX_ADAPT],
        label='1 group DLB', color='b', marker='o', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset2_vars_lb.SOLVER_T[1][:MAX_ADAPT],
        label='2 group DLB', color='g', marker='v', linestyle='--')
ax.plot(range(1, MAX_ADAPT+1), grpset3_vars_lb.SOLVER_T[1][:MAX_ADAPT],
        label='3 group DLB', color='m', marker='s', linestyle='--')
# ax.plot(range(1, MAX_ADAPT+1), grpset7_vars_lb.SOLVER_T[1][:MAX_ADAPT],
# label='7 group DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Adapt step', ylabel='Solver Time [s]')
ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()


# %%
# Plot the DLB time vs DOFs
fig, ax = plt.subplots()
ax.plot(grpset1_vars_lb.ALLDOFS[1][:MAX_ADAPT]*7, grpset1_vars_lb.LB_T[1]
        [:MAX_ADAPT], label='1 group set', color='b', marker='o', linestyle='--')
ax.plot(grpset2_vars_lb.ALLDOFS[1][:MAX_ADAPT]*7/2, grpset2_vars_lb.LB_T[1]
        [:MAX_ADAPT], label='2 group set', color='g', marker='v', linestyle='--')
ax.plot(grpset2_vars_lb.ALLDOFS[1][:MAX_ADAPT]*7/3, grpset2_vars_lb.LB_T[1]
        [:MAX_ADAPT], label='3 group set', color='m', marker='s', linestyle='--')
# ax.plot(grpset7_vars_lb.ALLDOFS[1][:MAX_ADAPT], grpset7_vars_lb.LB_T[1][:MAX_ADAPT],
# label='7 group set', color='r', marker='x', linestyle='--')

ax.set(xlabel='Av. DOFs per spatial node', ylabel='DLB Time [s]')
ax.legend()
fig.tight_layout()
if SAVE:
    fig_name = 'av_dofs_per_gset_vs_dlb_time'
    tikz_save(f'/home/gn/Dropbox/PhD/figures/{fig_name}.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')

# %%
# ## Sensitivity analysis for load balance tolerance
# %%
root_dir = '/home/gn/Code/Archer/tests/c5g7mox2D/exotic/new_tol_lb'
tol_list = []
fig, ax = plt.subplots()
markers = ['o', 'v', 's', 'x']
lb_tolerances = [1.00, 1.25, 1.50, 2.50]
linestyles = ['--', ':', '-.', '-']

for i, tol, mark, ln in zip(range(4), lb_tolerances, markers, linestyles):
    tol_list.append(ExtractLB())
    stat_f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_lb{0:.2f}_no_io'.format(tol)
    tol_list[i].get_run_stats(root_dir, stat_f)
    ax.plot(range(1, MAX_ADAPT+1), tol_list[i].WALL_T[1][:MAX_ADAPT],
            label=f't {tol}',
            marker=mark,
            linestyle=ln)

ax.set(xlabel='Adapt step', ylabel='Wall Time [s]')
ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()
if SAVE:
    fig_name = 'eigen_lb_tol_sensitivity_analysis'
    tikz_save(f'/home/gn/Dropbox/PhD/figures/{fig_name}.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


# %% [markdown]
# # Investigate what is happening with the iteration count in the power solves


# %%
root_dir = 'physor/group_full_flux_low_adapt_tol'
root_dir_lb = 'physor/group_full_lb_flux_low_adapt_tol'
stat_3f = 'rad_c5g7mox2D_3grp_hwn_reg_adapt_mf_no_io.Neutron.eig_iteration_convergence'
stat_2f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_no_io.Neutron.eig_iteration_convergence'
stat_1f = 'rad_c5g7mox2D_1grp_hwn_reg_adapt_mf_no_io.Neutron.eig_iteration_convergence'

stats_to_read = ['EigIteration', 'Keff', 'InnerIterations']
power_it_3gpr = {}
power_it_3gpr_lb = {}
power_it_2gpr = {}
power_it_2gpr_lb = {}
power_it_1gpr = {}
power_it_1gpr_lb = {}


def eigen_stat_file_slice(input_dict, max_adapt):
    """
        Slices the eigen iterations and all the other 
        values of an eig_iteration_convergence file up
        to the supplied adapt step
    """
    # Get all index positions in the dictionary which start with 1
    first_eigen_indices = [i for i, val in enumerate(
        input_dict['EigIteration']) if val == 1]

    # The data is already of correct size so return
    if max_adapt == len(first_eigen_indices):
        return input_dict

    # The adapt order requested is larger than the number of adapt steps present
    elif max_adapt > len(first_eigen_indices):
        max_adapt = len(first_eigen_indices) - 1

    # Slice all dictionary lists up to the max_adapt order
    for key in input_dict:
        input_dict[key] = input_dict[key][:first_eigen_indices[max_adapt]]

    return input_dict


# This will read the entire columns from the file which means that if we have
# data with varying orders of angular adapts, we will compare incorrect data
for i in stats_to_read:
    power_it_3gpr[i] = stat(f'{root_dir}/{stat_3f}')[i]['Value']
    power_it_3gpr_lb[i] = stat(f'{root_dir_lb}/{stat_3f}')[i]['Value']
    power_it_2gpr[i] = stat(f'{root_dir}/{stat_2f}')[i]['Value']
    power_it_2gpr_lb[i] = stat(f'{root_dir_lb}/{stat_2f}')[i]['Value']
    power_it_1gpr[i] = stat(f'{root_dir}/{stat_1f}')[i]['Value']
    power_it_1gpr_lb[i] = stat(f'{root_dir_lb}/{stat_1f}')[i]['Value']

# Go and count how many first eigen iterations we have in our data
# NOTE: Remember to do this for all the dictionaries
power_it_3gpr = eigen_stat_file_slice(power_it_3gpr, MAX_ADAPT)
power_it_3gpr_lb = eigen_stat_file_slice(power_it_3gpr_lb, MAX_ADAPT)

power_it_2gpr = eigen_stat_file_slice(power_it_2gpr, MAX_ADAPT)
power_it_2gpr_lb = eigen_stat_file_slice(power_it_2gpr_lb, MAX_ADAPT)

power_it_1gpr = eigen_stat_file_slice(power_it_1gpr, MAX_ADAPT)
power_it_1gpr_lb = eigen_stat_file_slice(power_it_1gpr_lb, MAX_ADAPT)

# %%
fig, ax = plt.subplots()
ax.plot(range(len(power_it_3gpr['InnerIterations'])),
        power_it_3gpr['InnerIterations'],
        label='3 group set', color='b', marker='s', markerfacecolor='none')
ax.plot(range(len(power_it_3gpr_lb['InnerIterations'])),
        power_it_3gpr_lb['InnerIterations'],
        label='3 group set DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Total Power iterations', ylabel='InnerIterations')
# ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()

# %%
fig, ax = plt.subplots()
ax.plot(range(len(power_it_2gpr['InnerIterations'])),
        power_it_2gpr['InnerIterations'],
        label='2 group set', color='b', marker='s', markerfacecolor='none')
ax.plot(range(len(power_it_2gpr_lb['InnerIterations'])),
        power_it_2gpr_lb['InnerIterations'],
        label='2 group set DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Total Power iterations', ylabel='InnerIterations')
# ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()
# %%
fig, ax = plt.subplots()
ax.plot(range(len(power_it_1gpr['InnerIterations'])),
        power_it_1gpr['InnerIterations'],
        label='1 group set', color='b', marker='s', markerfacecolor='none')
ax.plot(range(len(power_it_1gpr_lb['InnerIterations'])),
        power_it_1gpr_lb['InnerIterations'],
        label='1 group set DLB', color='r', marker='x', linestyle='--')

ax.set(xlabel='Total Power iterations', ylabel='InnerIterations')
# ax.set_xticks(range(1, MAX_ADAPT + 1))
# ax.set_yscale('log')
ax.legend()
fig.tight_layout()

# %% [markdown]
# # Plot all the power iterations in one graph for all the group sets
# %%
fig, ax = plt.subplots()
ax.plot(range(len(power_it_3gpr['InnerIterations'])),
        power_it_3gpr['InnerIterations'],
        label='3 group set', color='b', marker='s', markerfacecolor='none')
ax.plot(range(len(power_it_3gpr_lb['InnerIterations'])),
        power_it_3gpr_lb['InnerIterations'],
        label='3 group set DLB', color='r', marker='x', linestyle='--')

ax.plot(range(len(power_it_2gpr['InnerIterations'])),
        power_it_2gpr['InnerIterations'],
        label='2 group set', marker='o', markerfacecolor='none')
ax.plot(range(len(power_it_2gpr_lb['InnerIterations'])),
        power_it_2gpr_lb['InnerIterations'],
        label='2 group set DLB', marker='P', linestyle='--')

ax.plot(range(len(power_it_1gpr['InnerIterations'])),
        power_it_1gpr['InnerIterations'],
        label='1 group set', marker='D', markerfacecolor='none')
ax.plot(range(len(power_it_1gpr_lb['InnerIterations'])),
        power_it_1gpr_lb['InnerIterations'],
        label='1 group set DLB', color='cyan', marker='+', linestyle='--')

ax.set(xlabel='Total Power iterations', ylabel='InnerIterations')

ax.legend()
fig.tight_layout()

# %% [markdown]
# # Look at the individual linear solve iterations per power iteration per energy group
# %%
# Load the .stat files with the linear solver iterations
stat_3f = 'rad_c5g7mox2D_3grp_hwn_reg_adapt_mf_no_io.Neutron.iteration.stat'
stat_2f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_no_io.Neutron.iteration.stat'
stat_1f = 'rad_c5g7mox2D_2grp_hwn_reg_adapt_mf_no_io.Neutron.iteration.stat'

iter_count_3grp = stat(f'{root_dir}/{stat_3f}')['Iteration_Count']['Value']
iter_count_2grp = stat(f'{root_dir}/{stat_2f}')['Iteration_Count']['Value']
iter_count_1grp = stat(f'{root_dir}/{stat_1f}')['Iteration_Count']['Value']
iter_count_3grp_lb = stat(
    f'{root_dir_lb}/{stat_3f}')['Iteration_Count']['Value']
iter_count_2grp_lb = stat(
    f'{root_dir_lb}/{stat_2f}')['Iteration_Count']['Value']
iter_count_1grp_lb = stat(
    f'{root_dir_lb}/{stat_1f}')['Iteration_Count']['Value']

# We do 11 gauss-Seidel iterations through our energy groups
power_iter_stride = 11

# Keep only the data relevant up to the MAX_ADAPT, since we can have incomplete
# runs that terminated at some point in between and we can reshape those.
iter_count_3grp = iter_count_3grp[:len(power_it_3gpr['InnerIterations'])*power_iter_stride]
iter_count_2grp = iter_count_2grp[:len(power_it_2gpr['InnerIterations'])*power_iter_stride]
iter_count_1grp = iter_count_1grp[:len(power_it_1gpr['InnerIterations'])*power_iter_stride]
iter_count_3grp_lb = iter_count_3grp_lb[:len(power_it_3gpr_lb['InnerIterations'])*power_iter_stride]
iter_count_2grp_lb = iter_count_2grp_lb[:len(power_it_2gpr_lb['InnerIterations'])*power_iter_stride]
iter_count_1grp_lb = iter_count_1grp_lb[:len(power_it_1gpr_lb['InnerIterations'])*power_iter_stride]

# Place all the iterations in a list to make iterating easier
iter_count = [iter_count_1grp, iter_count_2grp, iter_count_3grp,
              iter_count_1grp_lb, iter_count_2grp_lb, iter_count_3grp_lb]

# Reshape all the iteration counts to for a single power iteration
for i in range(len(iter_count)):
    iter_count[i] = np.reshape(iter_count[i], (-1, power_iter_stride))

# Now we have 2D arrays for all the linear solver iterations per power iteration
# for all the adaptivity steps

# %% [markdown]
# ## Plot a heat map of the iteration count per power iteration, per adapt
# %%
rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(4, 10))
im = 0
images = []

for i in range(rows):
    for j in range(cols):
        images.append(axes[i, j].imshow(iter_count[im]))
        axes[i, j].label_outer()
        im += 1

# Find the min and max of all colors for use in setting the color scale.
vmin = min([np.amin(image) for image in iter_count])
vmax = max([np.amax(image) for image in iter_count])
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=.1)

# %%
