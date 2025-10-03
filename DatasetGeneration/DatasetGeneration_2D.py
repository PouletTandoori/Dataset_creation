from ModelGenerator_modified import (Sequence, Stratigraphy,
                            Property, Lithology, ModelGenerator, Deformation)
import os

from GeoDataset_modified import GeoDataset
from SeismicGenerator_modified import SeismicGenerator, plot_shotgather
from SeismicGenerator_modified import SeismicAcquisition, SeismicAcquisition2
from GraphIO_modified import Vsdepth, ShotGather, Vpdepth, Dispersion, DispersionCurves
import argparse
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
from tabulate import tabulate
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#define parser
parser = argparse.ArgumentParser()
parser.add_argument('-tr_s','--trainsize',type=int,default=10,
                    help='Number of files to be generated in the training folder,'
                         '\n validation and test folders will have 30% of this number')
parser.add_argument('-data','--dataset_name',type=str,default='Halton_debug',
                    help='Name of the dataset, ex: Halton_debug')
parser.add_argument('--Halton_seq', type=bool,
                    default=False,
                    help='Start from a given Halton sequence, instead of starting from zero')
parser.add_argument('--GPU','-gpu', type=str, nargs='+',
                    default=['2'],
                    help='GPU ids to use for training (ex: --GPU 0 1)')
args = parser.parse_args()

# set the GPU ids to use for training:

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.GPU)
print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
# IDs of GPU to use:
nproc = os.environ["CUDA_VISIBLE_DEVICES"]  # Number of GPUs to use for training

#verify if Datasets folder exists, if not create it:
if not os.path.exists('Datasets/'):
    os.mkdir('Datasets/')

if args.Halton_seq is False:
    # erase the previous dataset
    if os.path.exists(f'Datasets/{args.dataset_name}/'):
        os.system(f'rm Datasets/{args.dataset_name}/*/*')
        print(f'rm Datasets/{args.dataset_name}/*/*')
        print('---------------------------------------------------')
        print('\n')


#create folder for the figures if does not exist:
path_figures=f'../figures/{args.dataset_name}'
if not os.path.exists(path_figures):
    os.mkdir(path_figures)
    print('Creating folder for the figures')
    print('---------------------------------------------------')
    print('\n')

#Define parameters for the dataset:
train_size= args.trainsize
validate_size= int(train_size*0.3)
test_size= int(train_size*0.3)

#Define parameters for aquisition geometry:
length_profile=96
dh = 0.5                                            # Grid spacing in meters# Length of the line of receivers: 96m (=m*Zmax with 1<m<3)
minimal_offset=length_profile*dh                    # Minimal offset: generally L/4
bonus_cells = 100                                     # Number of bonus cells to avoid disposal to close to the boundary
Profile=length_profile+minimal_offset+bonus_cells   # Length of the profile: 96+24=120m
number_receivers=96                                 # Number of receivers: 96
receiver_spacing=1                                  # Receiver spacing: 1m (Zmin/k with 0.3<k<1.0)
Zmax=50                                             # Maximum depth of the profile: 50m

#Define parameters for the modelisation of the seismic data:
nab = 128                                           # Number of padding cells for the absorbing boundary
peak_freq = 10                                      # Peak frequency of the wavelet in Hertz
df = 2                                              # Frequency deviation for randomness


dt = 0.0001                                        # Time sampling interval in seconds
resampling = 15                                     # Resampling factor for the time axis
NbCellsX= int(Profile/dh)                           # Number of grid cells in x direction (domain size + boundary)
NbCellsZ= int(Zmax/dh)                              # Number of grid cells in z direction (domain size + boundary)



class ViscoElasticModel(ModelGenerator):

    def __init__(self,):
        super().__init__()
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = dh
        # Number of grid cells in X direction.
        self.NX = NbCellsX
        # Number of grid cells in Z direction.
        self.NZ = NbCellsZ

        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_min = 2/self.dh
        # Maximum thickness of a layer (in grid cells).
        self.layer_dh_max = 25/self.dh
        # Minimum number of layers.
        self.layer_num_min = 4
        # Fix the number of layers if not 0.
        self.num_layers = 0

        # If true, first layer dip is 0.
        self.dip_0 = True
        # Maximum dip of a layer.
        self.dip_max = 0
        # Maximum dip difference between two adjacent layers.
        self.ddip_max = 0

        # Change between two layers.
        # Add random noise two a layer (% or velocity).
        self.max_texture = 0
        # Range of the filter in x for texture creation.
        self.texture_xrange = 0
        # Range of the filter in z for texture creation.
        self.texture_zrange = 0
        # Zero-lag correlation between parameters, same for each
        self.corr = 0.6

        # Minimum fault dip.
        self.fault_dip_min = 0
        # Maximum fault dip.
        self.fault_dip_max = 45
        # Minimum fault displacement.
        self.fault_displ_min = 0
        # Maximum fault displacement.
        self.fault_displ_max = 10
        # Bounds of the fault origin location.
        self.fault_x_lim = [0, self.NX]
        self.fault_y_lim = [0, self.NZ]
        # Maximum quantity of faults.
        self.fault_nmax = 1
        # Probability of having faults.
        self.fault_prob = 0.0

        self.thick0min = None
        self.thick0max = 50
        self.layers = None

        self._properties = None
        self._stratigraphy = None

    def VpVs_from_VpandVs(self, vp_min, Vs_min, vp_max, Vs_max):
        """
        Compute Vp/Vs min and max ratios from Vp and Vs.
        """
        vpvs_min = vp_min / Vs_max
        vpvs_max = vp_max / Vs_min
        return vpvs_min, vpvs_max

    def Summary_lithologies(self, strati, properties):
        print("Summary of the lithologies:\n")

        # Collect data for the table
        table_data = []
        for seq in strati.sequences:
            for lith in seq.lithologies:
                vp_min = next(prop.min for prop in lith.properties if prop.name == "vp")
                vp_max = next(prop.max for prop in lith.properties if prop.name == "vp")
                vs_min = next(prop.min for prop in lith.properties if prop.name == "vs")
                vs_max = next(prop.max for prop in lith.properties if prop.name == "vs")
                vpvs_min = next(prop.min for prop in lith.properties if prop.name == "vpvs")
                vpvs_max = next(prop.max for prop in lith.properties if prop.name == "vpvs")
                rho_min = next(prop.min for prop in lith.properties if prop.name == "rho")
                rho_max = next(prop.max for prop in lith.properties if prop.name == "rho")
                q_min = next(prop.min for prop in lith.properties if prop.name == "q")
                q_max = next(prop.max for prop in lith.properties if prop.name == "q")

                # Add a row for the lithology
                table_data.append([
                    lith.name, vp_min, vp_max,
                    vs_min, vs_max,
                    vpvs_min, vpvs_max,
                    rho_min, rho_max,
                    q_min, q_max
                ])

        # Define headers
        headers = [
            "Lithology", "vp_min", "vp_max",
            "vs_min", "vs_max",
            "vpvs_min", "vpvs_max",
            "rho_min", "rho_max",
            "q_min", "q_max"
        ]

        # Print the table
        print(tabulate(table_data, headers=headers, floatfmt=".2f", tablefmt="fancy_grid"))
        print('---------------------------------------------------')
        print('\n')

    def build_stratigraphy(self):
        lithologies = {}

        '''# Organic soils
        name = "Organic soils"
        vp = Property("vp", vmin=300, vmax=700)
        vs = Property("vs", vmin=100, vmax=300)
        print('Organic soils:')
        print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=500, vmax=1500)
        q = Property("q", vmin=5, vmax=20)
        organic_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        # Dry sands
        name = "Dry Sands"
        vp = Property("vp", vmin=400, vmax=1200)
        vs = Property("vs", vmin=100, vmax=500)
        #print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        #print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=1700, vmax=1900)
        q = Property("q", vmin=13, vmax=63)
        dry_sands = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Wet sands
        name = "Wet sands"
        vp = Property("vp", vmin=1500, vmax=2000)
        vs = Property("vs", vmin=400, vmax=600)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1800, vmax=2000)
        q = Property("q", vmin=13, vmax=63)
        wet_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # silts
        name = "Silts"
        vp = Property("vp", vmin=1400, vmax=2100)
        vs = Property("vs", vmin=300, vmax=800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1300, vmax=2200)
        q = Property("q", vmin=13, vmax=63)
        silts = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Clay
        name = "Clay"
        vp = Property("vp", vmin=1100, vmax=2500)
        vs = Property("vs", vmin=80, vmax=800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1200, vmax=2000)
        q = Property("q", vmin=7, vmax=14)
        clay = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        '''#tills
        name = "Tills"
        vp = Property("vp", vmin=1600, vmax=3100)
        vs = Property("vs", vmin=300, vmax=1100)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1600, vmax=2100)
        q = Property("q", vmin=256, vmax=430)
        tills = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# sandstone
        name = "Sandstone" #grès
        vp = Property("vp", vmin=2000, vmax=3500)
        vs = Property("vs", vmin=800, vmax=1800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2100, vmax=2400)
        q = Property("q", vmin=70, vmax=150)
        sandstone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Dolomite
        name = "Dolomite"  # dolomie
        vp = Property("vp", vmin=2500, vmax=6500)
        vs = Property("vs", vmin=1900, vmax=3600)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2300, vmax=2900)
        q = Property("q", vmin=100, vmax=600)
        dolomite = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Limestone
        name = "Limestone"  # calcaire
        vp = Property("vp", vmin=2300, vmax=2600)
        vs = Property("vs", vmin=1100, vmax=1300)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2600, vmax=2700)
        q = Property("q", vmin=100, vmax=600)
        limestone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Shale
        name = "Shale"
        vp = Property("vp", vmin=2000, vmax=5000)
        vs = Property("vs", vmin=1000, vmax=2000)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1700, vmax=3300)
        q = Property("q", vmin=10, vmax=70)
        shale = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''


        # set everything to zero if you want a tabular model.
        deform = Deformation(max_deform_freq=0.00,
                             min_deform_freq=0.0000,
                             amp_max=0,
                             max_deform_nfreq=0,
                             prob_deform_change=0)


        fine_sediments_seq = Sequence(name="fine_sediments_seq",
                                    lithologies=[dry_sands, wet_soils, silts, clay],
                                    thick_min=self.NZ,
                                    thick_max=self.NZ,
                                    deform=deform,
                                    skip_prob=0.0,
                                    ordered=False,
                                    accept_decrease=0.2,
                                    nmax=15,
                                    halton_seq=self.Halt_seq,
                                    nlayers=self.nlayer)

        # dry soil, saturated soil, altered rock, granite
        sequences = [fine_sediments_seq]
        strati = Stratigraphy(sequences=sequences,Halton_seq=self.Halt_seq)
        self._stratigraphy = strati


        properties = strati.properties()


        return strati, properties


    def generate_model(self, seed=235):
        props2D, layerids, layers,Halton_reduced = super().generate_model(seed=seed,Halt_seq=self.Halt_seq)
        props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers,Halton_reduced


#Display the summary of the lithologies:
model=ViscoElasticModel()
#model.Summary_lithologies(model.stratigraphy, model.properties)

#define MyAcquisition class using SeismicAcquisition class from SeismicGenerator.py
class MyAcquisition2(SeismicAcquisition2):
    '''
    We want to make 1.5s long seismograms with 0.02 ms sampling rate.
    If we want a minimum depth investigation of 2m and a maximum depth investigation of 50m, we need to have a grid spacing of 2/8 m = 0.25m.
    We want to have 50m long line of receivers with 96 receivers spaced by 1m.

    We will create a grid of 100m long * 50m depth grid cells with a grid spacing of 0.25m.

    '''

    def __init__(self, dh: float = dh, nx: int = NbCellsX, nz: int = NbCellsZ):
        super().__init__(dh=dh, nx=nx, nz=nz)

        # Time sampling for seismogram (in seconds).
        self.dt = dt
        # Number of times steps.
        self.NT = int(1.5/dt)  # around 1.5 sec
        # Peak frequency of input wavelet (in Hertz).
        self.peak_freq = peak_freq
        self.configuration = 'inline'
        # Minimum position of receivers (-1 = minimum of grid).
        self.gmin = None
        # Maximum position of receivers (-1 = maximum of grid).
        self.gmax = None
        self.resampling = resampling  # ex: choose 10 to divide time axis by 10
        self.singleshot = False

class MyAcquisition(SeismicAcquisition):
    '''
    We want to make 1.5s long seismograms with 0.02 ms sampling rate.
    If we want a minimum depth investigation of 2m and a maximum depth investigation of 50m, we need to have a grid spacing of 2/8 m = 0.25m.
    We want to have 50m long line of receivers with 96 receivers spaced by 1m.

    We will create a grid of 100m long * 50m depth grid cells with a grid spacing of 0.25m.

    '''

    def __init__(self, dh: float = dh, nx: int = NbCellsX, nz: int = NbCellsZ):
        super().__init__(dh=dh, nx=nx, nz=nz)


        #-------------------------------------------------------#
        # Number of padding cells of absorbing boundary.
        self.nab = nab

        self.dh = dh  # grid spacing, in meters
        self.nx = nx + 2 * self.nab  # number of grid cells in x direction
        self.nz = nz + 2 * self.nab  # number of grid cells in z direction

        # Whether free surface is turned on the top face.
        self.fs = True
        # Time sampling for seismogram (in seconds).
        self.dt = dt
        # Number of times steps.
        self.NT = int(1.5 / self.dt)
        # Peak frequency of input wavelet (in Hertz).
        self.peak_freq = peak_freq  # could be higher
        # Source wave function selection.
        self.wavefuns = [1]
        # Frequency of source peak_freq +- random(df).
        self.df = 2
        # Delay of the source.
        self.tdelay = self.dt * 5  # 2.0 / (self.peak_freq - self.df)
        # Resampling of the shots time axis.
        self.resampling = resampling  # ex: choose 10 to divide time axis by 10
        # Depth of sources (m).
        self.source_depth = 0
        # Depth of receivers (m).
        self.receiver_depth = 0
        # Receiver interval in meters
        self.dg = 1
        # Source interval (in 2D).
        self.ds = 2
        # Minimum position of receivers (-1 = minimum of grid).
        self.gmin = None
        # Maximum position of receivers (-1 = maximum of grid).
        self.gmax = None
        self.minoffset = self.dg  # (nx - 2 * self.nab)/4
        # Integer used by SeisCL for pressure source (100) or force in z (2).
        self.sourcetype = 2
        # Integer used by SeisCL indicating which type of recording. Either
        # 2) pressure or 1) velocities.
        self.rectype = 1  # if we choose velocities, then we obtain 2 shotgathers instead of 1
        # Absorbing boundary type:
        # 1) CPML or 2) absorbing layer of Cerjan.
        self.abs_type = 2

        self.singleshot = True
        # Whether to fill the surface with geophones or to use inline spread.
        # Either `'full'` or `'inline'`.
        self.configuration = 'inline'
        # Number of attenuation mechanism (L=0 elastic)
        self.L = 0
        # Frequencies of the attenuation mechanism
        self.FL = np.array(15)

    def set_rec_src(self):
        #print('nab = ', self.nab)

        dg = self.dg / self.dh
        # convert into grid points
        ng = 96  # Quantity of geophones.

        # Add receiver.
        if self.gmin:
            gmin = self.gmin
        else:
            gmin = self.nab + self.minoffset

        if self.gmax:
            gmax = self.gmax
        else:
            gmax = gmin + ng * dg

        # Add sources.
        sx = [gmin * self.dh - self.minoffset * self.dh]

        # Set source.
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, len(sx))

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)], axis=0)

        # Set receivers
        gx0 = np.arange(gmin, gmax, dg) * self.dh
        gx = np.concatenate([gx0 for _ in sx], axis=0)
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        gz = np.full_like(gx, self.receiver_depth)
        gid = np.arange(0, len(gx))

        rec_pos = np.stack([gx,
                            np.zeros_like(gx),
                            gz,
                            gsid,
                            gid,
                            np.full_like(gx, 2),
                            np.zeros_like(gx),
                            np.zeros_like(gx)], axis=0)

        return src_pos, rec_pos

    def plot_acquisition_geometry(self, path=None):
        src_pos, rec_pos = self.set_rec_src()
        qty_srcs = src_pos.shape[1]
        src_x, _, _, src_id, _ = src_pos
        rec_x, _, _, rec_src_id, _, _, _, _ = rec_pos

        x_max = self.nx * self.dh
        z_max = self.nz * self.dh

        # Configuration du graphique
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- Tracé du modèle (fond) ---
        model_rect = patches.Rectangle((0, 0), x_max, z_max, color='lightgray', alpha=0.3)
        ax.add_patch(model_rect)

        # --- Tracé des sources ---
        ax.scatter(src_x, [0] * len(src_x), marker='x', s=60, color='red', label='Sources')

        # --- Tracé des récepteurs ---
        ax.scatter(rec_x, [0.5] * len(rec_x), marker='v', s=40, color='blue', label='Receivers')

        # --- Réglage des axes ---
        ax.set_xlim(src_x[0] - 10, rec_x[-1] + 10)  # Ajuster les limites des axes x
        ax.set_ylim(-1, z_max / 3)
        ax.set_xlabel("Position horizontale (m)")
        ax.set_ylabel("Profondeur (m)")
        ax.invert_yaxis()  # Pour que 0 soit en haut (comme une coupe géologique)
        ax.grid(True, linestyle="--", alpha=0.4)

        # --- Légende ---
        ax.legend(loc='upper right', fontsize=12)

        # --- Titre ---
        ax.set_title("Géométrie d'acquisition sismique", fontsize=14, fontweight='bold')

        # --- Sauvegarde / affichage ---
        if path is not None:
            plt.savefig(f'{path}/acquisition_geometry.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.close()


#plot the aquisition geometry:
aquire=MyAcquisition2()


class SimpleDataset(GeoDataset):
    name = args.dataset_name

    def set_dataset(self, *args, **kwargs):
        model=ViscoElasticModel()
        acquire = MyAcquisition2()
        print('Acquisition.NT :',acquire.NT)

        physic = SeismicGenerator(acquire=acquire)
        graphios = {ShotGather.name: ShotGather(model=model, acquire=acquire),
                    Dispersion.name: Dispersion(model=model, acquire=acquire,
                                                cmax=900, cmin=50, fmin=1, fmax=50),
                    DispersionCurves.name: DispersionCurves(model=model, acquire=acquire,
                                                            cmax=900, cmin=50, fmin=1, fmax=50),
                    Vsdepth.name: Vsdepth(model=model, acquire=acquire),
                    Vpdepth.name: Vpdepth(model=model, acquire=acquire)}
        for name in graphios:
            graphios[name].train_on_shots = True

        return model, physic, graphios


dataset = SimpleDataset(trainsize=train_size, validatesize=validate_size, testsize=test_size,
                        toinputs={ShotGather.name: ShotGather.name},
                        tooutputs={Vsdepth.name: Vsdepth.name, Vpdepth.name: Vpdepth.name})

#create ModelAnimated:
print('Creating model visualisation:')
#dataset.model.animated_dataset(nframes=min(args.trainsize+args.validatesize+args.testsize,50),path=path_figures)
#generate dataset:
print('\n')
print('---------------------------------------------------')
print('Generating dataset:')
#nproc = id selected GPUs
dataset.generate_dataset(nproc=nproc)
print('Dataset created')
print('---------------------------------------------------')
print('\n')

print('Erasing lock files')
# erase all .lock files in train, validate and test folders:
os.system(f'rm Datasets/{args.dataset_name}/train/*.lock')
os.system(f'rm Datasets/{args.dataset_name}/validate/*.lock')
os.system(f'rm Datasets/{args.dataset_name}/test/*.lock')



#smallest between 5 and the sum of the sizes of the training, validation, and test sets
nb_examples = min(5, args.trainsize + validate_size + test_size)
# Collect multiple examples from the dataset
examples = [dataset.get_example(phase="train") for _ in range(nb_examples)]
# Unpack the examples into separate lists for inputs, labels, weights, and filenames
inputspre_list, labelspre_list, weightspre_list, filename_list = zip(*examples)



#change direction to /home/rbertille/data/pycharm/ViT_project/pycharm_ViT to make post processing
print('Post processing:')
os.chdir('/home/rbertille/data/pycharm/ViT_project/pycharm_ViT')
subprocess.run(['python', 'verify_data.py', '--dataset_name', args.dataset_name, '--clean', 'True'])
subprocess.run(['python','DistributionDataset.py','--dataset_name',args.dataset_name])

os.chdir('/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/DatasetGeneration')
aquire.plot_acquisition_geometry(path=path_figures)
#plot the example:
plot_shotgather(inputspre_list, labelspre_list,dz=dh,dt=dt,resample=resampling,path=path_figures, nb_examples=nb_examples)