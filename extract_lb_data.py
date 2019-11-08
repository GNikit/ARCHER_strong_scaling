
import os
import sys
import numpy as np
from scipy import stats

try:
    from fluidity_tools import stat_parser as stat
except ImportError:
    # try again by adding the path "../python" relative to testharness' own location to sys.path
    head, tail = os.path.split(sys.argv[0])
    # python_path = os.path.abspath(os.path.join(head, '..', 'python'))
    sys.path.append('/home/gn/Code/fetch2012/fluidity/python')
    from fluidity_tools import stat_parser as stat


class ExtractLB:
    def __init__(self):
        self.WALL_T = {}
        self.LB_T = {}
        self.SOLVER_T = {}
        self.RIEMMAN_T = {}
        self.CDOFS = {}
        self.DDOFS = {}
        self.ALLDOFS = {}
        self.Nnodes = 1

    def get_run_stats(self, dir, log_name, cores=None,
                      solver_t=False, riemman_t=False):
        """
            Extracts the Load balancer, solver and CDOFs
            out of the stat files, crawling directories following the name
            coreXXX/rad_radiant_noio.e

            If multiple group sets exist in the file, then all of them will be 
            read, using the format of XXX_gset_G, where X is the cpu and G the
            group set number
        """
        # Store the directory before you start
        root_dir = os.getcwd()
        STAT_F = f'{log_name}.Neutron.output_quantities.stat'

        # If we don't have multiple cores to loop through just set the core
        # count to 1
        no_cores_dir = False
        if cores is None:
            no_cores_dir = True
            cores = [1]

        for cpu in cores:
            # If we are crawling through multiple core directories
            # Change to that core dir
            if no_cores_dir:
                os.chdir(f'{dir}')
            else:
                os.chdir(f'{dir}/core{cpu}')

            self.LB_T[cpu] = stat(STAT_F)['RadiantLoadBalanceTime']['Value']
            self.WALL_T[cpu] = stat(STAT_F)['ElapsedWallTime']['Value']

            if solver_t:
                self.SOLVER_T[cpu] = stat(STAT_F)['RadiantSolveTime']['Value']

            if riemman_t:
                self.RIEMMAN_T[cpu] = stat(
                    STAT_F)['RadiantCalcRiemmanMatsTime']['Value']

            self.get_average_dofs(STAT_F, cpu)

        # Change back to the initial directory
        os.chdir(root_dir)

        return self.WALL_T, self.LB_T, self.CDOFS

    def get_average_dofs(self, STAT_F, cpu):
        """
            NOTE: we are not normalising with the number of
                   energy groups per group set to give the 
                   total number of DOFs. The user will have 
                   to multiply with the ratio of energy groups/group sets
                   to get the total DOFs

            NOTE 2: This will not work with spatial adaptivity
        """
        # Try and get the number of nodes when one group set is used
        try:
            self.Nnodes = stat(STAT_F)['NumberOfNodes']['Value'][-1]
        except KeyError:
            self.Nnodes = stat(STAT_F)['NumberOfContNodes_gset_1']['Value'][-1]

        g_set = 0
        # Try to get the DOFs for when one group set is used
        try:
            self.CDOFS[cpu] = stat(
                STAT_F)['ContinuousDOF_per_group']['Value']
            self.DDOFS[cpu] = stat(
                STAT_F)['DiscontinuousDOF_per_group']['Value']
        # We have multiple group sets. Lets extract them recursively
        except KeyError:
            # Start reading the group sets
            g_set = 1
            while True:
                try:
                    self.CDOFS[f'{cpu}_gset_{g_set}'] = stat(
                        STAT_F)[f'ContinuousDOF_per_group_gset_{g_set}']['Value']
                    self.DDOFS[f'{cpu}_gset_{g_set}'] = stat(
                        STAT_F)[f'DiscontinuousDOF_per_group_gset_{g_set}']['Value']
                    g_set += 1
                # We run out of group sets to read so exit
                except KeyError:
                    break

        # Both the CDOFs and the DDOFs have the same keys, therefore simply
        # add the two dictionaries.
        # We know how many group sets we have, so add the DOFs of all them
        if g_set > 1:
            temp = [0] * len(self.CDOFS[f'{cpu}_gset_1'])
            for g in range(1, g_set):
                temp += self.CDOFS[f'{cpu}_gset_{g}'] + \
                    self.DDOFS[f'{cpu}_gset_{g}']
            self.ALLDOFS[cpu] = temp

        else:
            self.ALLDOFS[cpu] = self.CDOFS[cpu] + self.DDOFS[cpu]

        # Normalise with the the number of spatial nodes
        self.ALLDOFS[cpu] /= self.Nnodes

        return self.ALLDOFS[cpu]

    def get_partition_stats(self, dir, log_name, max_adapt, num_cores):
        """
        Get the mean and mode partition sizes for each adapt step
        for a given number of processors.

        len(mean_partition_sizes) : max_adapt
        len(mode_partition_sizes) : max_adapt
        len(nodes): max_adapt * num_cores
        """

        # Store the directory before you start
        root_dir = os.getcwd()
        os.chdir(f'{dir}/core{num_cores}')

        # First load the size of partitions from log files
        # You need to checkout archer_loadbalancer branch
        nodes = []
        for i in range(num_cores):
            nodes.extend(np.loadtxt(f'{log_name}.log_{i}', usecols=(1)))

        mean_partition_sizes = []
        mode_partition_sizes = []
        for adapt in range(max_adapt):
            # Gets the average partition size, which should remain constant
            # unless the elements present in the spatial mesh change
            mean_partition_sizes.append(
                sum(nodes[adapt::max_adapt])/len(nodes[adapt::max_adapt]))

            # Calculates the Mode of the partition sizes
            # TODO: write algorithm for multiple modes of an array
            mode_partition_sizes.append(stats.mode(nodes[adapt::max_adapt])[0])

        # Convert to numpy arrays
        mean_partition_sizes = np.asarray(mean_partition_sizes)
        mode_partition_sizes = np.asarray(mode_partition_sizes)
        nodes = np.asarray(nodes)

        # Return to original directory
        os.chdir(root_dir)

        return mean_partition_sizes, mode_partition_sizes, nodes

    def get_halo_stats(self, dir, log_name, max_adapt, num_cores):
        """
        Get the mean and mode of for halo sizes 
        for a given number of processors at every adapt step

        len(mean_halo_sizes) : max_adapt
        len(mode_halo_sizes) : max_adapt
        len(halos): max_adapt * num_cores
        """

        # Store the directory before you start
        root_dir = os.getcwd()
        os.chdir(f'{dir}/core{num_cores}')

        halos = []
        for i in range(num_cores):
            halos.extend(np.loadtxt(f'{log_name}.log_{i}', usecols=(3)))

        mean_halo_sizes = []
        mode_halo_sizes = []
        for adapt in range(max_adapt):
            # Gets the average partition size, which should remain constant
            # unless the elements present in the spatial mesh change
            mean_halo_sizes.append(
                sum(halos[adapt::max_adapt])/len(halos[adapt::max_adapt]))

            # Calculates the Mode of the partition sizes
            # TODO: write algorithm for multiple modes of an array
            mode_halo_sizes.append(stats.mode(halos[adapt::max_adapt])[0])

        # Convert to numpy array
        mean_halo_sizes = np.asarray(mean_halo_sizes)
        mode_halo_sizes = np.asarray(mode_halo_sizes)
        halos = np.asarray(halos)

        # Return to original directory
        os.chdir(root_dir)

        return mean_halo_sizes, mode_halo_sizes, halos

    def get_node_to_halo_ratio_stats(self, dir, log_name, max_adapt, num_cores):
        __, __, halos = self.get_halo_stats(
            dir, log_name, max_adapt, num_cores)
        __, __, nodes = self.get_partition_stats(
            dir, log_name, max_adapt, num_cores)

        ratio = nodes/halos

        mean_ratio = []
        mode_ratio = []
        for adapt in range(max_adapt):
            mean_ratio.append(
                sum(ratio[adapt::max_adapt])/len(ratio[adapt::max_adapt]))
            mode_ratio.append(stats.mode(ratio[adapt::max_adapt])[0])

        return mean_ratio, mode_ratio, ratio

    # def get_node_to_halo_3d(self, dir, )

    def get_strong_scaling(self, cores):
        """ Pass a list for the number of cores used in the study.
            And assuming you have already loaded the data
            you will generate the strong scaling performance
        """
        return [self.strong_scaling(self.WALL_T[min(cores)][-1], min(cores), self.WALL_T[cpu][-1], cpu) for cpu in cores]

    @staticmethod
    def strong_scaling(t0, c0, tn, cn):
        """
            Takes the time and number of cores for 2 simulations and returns 
        """
        return t0 / (cn/c0 * tn) * 100


# Example of how you save tikz images
# tikz_save('/home/gn/Dropbox/PhD/Posters/ESE_2019/figures/dogleg_speedup.tikz',
#             figureheight='\\figureheight',
#             figurewidth='\\figurewidth')
