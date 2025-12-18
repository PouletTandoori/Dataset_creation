from GeoFlow.Physics import Physic

from ModelGenerator_modified import (Sequence, Stratigraphy,
                            Property, Lithology, ModelGenerator, Deformation)
import os

tmpdir = "/userdata/u/rbertille/tmp_remy"
os.environ["TMPDIR"] = tmpdir
os.environ["OMPI_MCA_orte_tmpdir_base"] = tmpdir

# visible devices = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# ID of GPU:
nproc = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

from SeismicGenerator_modified import SeismicGenerator, plot_shotgather
from SeismicGenerator_modified import SeismicAcquisition, SeismicAcquisition2
import argparse
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
import numpy as np
from Halton_Sequence import HaltonSequenceGenerator
from GraphIO_modified import ShotGather
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import deepwave
from math import sqrt

#define parser
parser = argparse.ArgumentParser()
parser.add_argument('-tr_s','--trainsize',type=int,default=1,
                    help='Number of files to be generated in the training folder,'
                         '\n validation and test folders will have 30% of this number')
parser.add_argument('-data','--dataset_name',type=str,default='test_deepwave',
                    help='Name of the dataset, ex: Halton_Dataset4')
parser.add_argument('--Halton_seq', type=bool,
                    default=False,
                    help='Start from a given Halton sequence, instead of starting from zero')
args = parser.parse_args()

# set the GPU ids to use for training:

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
path_figures=f'figures/{args.dataset_name}'
print('figures path:',os.path.abspath(path_figures))
if not os.path.exists(path_figures):
    os.mkdir(path_figures)
    print('Creating folder for the figures here:', os.path.abspath(path_figures))
    print('---------------------------------------------------')
    print('\n')

#Define parameters for the dataset:
train_size= args.trainsize
validate_size= 0
test_size= 0

#Define parameters for the modelisation of the seismic data:
nab = 128                                           # Number of padding cells for the absorbing boundary
peak_freq = 15                                      # Peak frequency of the wavelet in Hertz
df = 2                                              # Frequency deviation for randomness

#Define parameters for aquisition geometry:
length_profile=96
dh = 0.5                                              # Grid spacing in meters# Length of the line of receivers: 96m (=m*Zmax with 1<m<3)
minimal_offset=length_profile*dh                    # Minimal offset: generally L/4
bonus_cells = 200                                   # Number of bonus cells to avoid disposal to close to the boundary
Profile=length_profile+minimal_offset+2*nab*dh + 20 # Length of the profile: 96+ 48 + 128 + 20 = 196+96= 292 cells
number_receivers=96                                 # Number of receivers: 96
receiver_spacing=1                                  # Receiver spacing: 1m (Zmin/k with 0.3<k<1.0)
Zmax=50                                             # Maximum depth of the profile: 50m

dt = 0.00004                                        # Time sampling interval in seconds
resampling = 1                                     # Resampling factor for the time axis
NbCellsX= int(Profile/dh)                           # Number of grid cells in x direction (domain size + boundary) = 292 /0.5 = 584
NbCellsZ= int(Zmax/dh) + nab                        # Number of grid cells in z direction (domain size + boundary) = 50/0.5 + 128 = 228



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
        self.layer_num_min = 10
        # Maximum number of layers
        self.num_layers_max = 20
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
        self.fault_prob = 0.1

        self.thick0min = None
        self.thick0max = 50
        self.layers = None

        self._properties = None
        self._stratigraphy = None

        # Generate Halton sequence for the model parameters
        intervals = [(0, 1)] * 7  # 6 dimensions between 0 and 1
        Halton_sequence = HaltonSequenceGenerator(intervals, integer=False)
        n_samples = self.num_layers_max * int(args.trainsize * 3)
        Halton_seq = np.array(Halton_sequence.generate(n_samples))
        keys = ['Nb_layers', 'Ordered', 'thickness', 'litho', 'Vp', 'Vs', 'Q']
        self.Halt_seq = {keys[i]: Halton_seq[:, i] for i in range(len(keys))}


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
        #print(tabulate(table_data, headers=headers, floatfmt=".2f", tablefmt="fancy_grid"))
        #print('---------------------------------------------------')
        #print('\n')

    def build_stratigraphy(self):
        lithologies = {}

        # Organic soils
        name = "Organic soils"
        vp = Property("vp", vmin=300, vmax=700,texture=30)
        vs = Property("vs", vmin=100, vmax=300,texture=30)
        print('Organic soils:')
        print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=500, vmax=1500)
        q = Property("q", vmin=5, vmax=20)
        organic_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Dry sands
        name = "Dry Sands"
        vp = Property("vp", vmin=400, vmax=1200,texture=30)
        vs = Property("vs", vmin=100, vmax=500,texture=30)
        #print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        #print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=1700, vmax=1900)
        q = Property("q", vmin=13, vmax=63)
        dry_sands = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Wet sands
        name = "Wet sands"
        vp = Property("vp", vmin=1500, vmax=2000,texture=30)
        vs = Property("vs", vmin=400, vmax=600,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1800, vmax=2000)
        q = Property("q", vmin=13, vmax=63)
        wet_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # silts
        name = "Silts"
        vp = Property("vp", vmin=1400, vmax=2100,texture=30)
        vs = Property("vs", vmin=300, vmax=800,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1300, vmax=2200)
        q = Property("q", vmin=13, vmax=63)
        silts = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Clay
        name = "Clay"
        vp = Property("vp", vmin=1100, vmax=2500,texture=30)
        vs = Property("vs", vmin=80, vmax=800,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1200, vmax=2000)
        q = Property("q", vmin=7, vmax=14)
        clay = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        #tills
        name = "Tills"
        vp = Property("vp", vmin=1600, vmax=3100,texture=30)
        vs = Property("vs", vmin=300, vmax=1100,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1600, vmax=2100)
        q = Property("q", vmin=256, vmax=430)
        tills = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # sandstone
        name = "Sandstone" #grès
        vp = Property("vp", vmin=2000, vmax=3500,texture=30)
        vs = Property("vs", vmin=800, vmax=1800,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2100, vmax=2400)
        q = Property("q", vmin=70, vmax=150)
        sandstone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Dolomite
        name = "Dolomite"  # dolomie
        vp = Property("vp", vmin=2500, vmax=6500,texture=30)
        vs = Property("vs", vmin=1900, vmax=3600,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2300, vmax=2900)
        q = Property("q", vmin=100, vmax=600)
        dolomite = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Limestone
        name = "Limestone"  # calcaire
        vp = Property("vp", vmin=2300, vmax=2600,texture=30)
        vs = Property("vs", vmin=1100, vmax=1300,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2600, vmax=2700)
        q = Property("q", vmin=100, vmax=600)
        limestone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Shale
        name = "Shale"
        vp = Property("vp", vmin=2000, vmax=5000,texture=30)
        vs = Property("vs", vmin=1000, vmax=2000,texture=30)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1700, vmax=3300)
        q = Property("q", vmin=10, vmax=70)
        shale = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])


        # set everything to zero if you want a tabular model.
        deform = Deformation(max_deform_freq=0.02,
                             min_deform_freq=0.0001,
                             amp_max=25,
                             max_deform_nfreq=40,
                             prob_deform_change=.3,
                             cumulative=True)


        fine_sediments_seq = Sequence(name="fine_sediments_seq",
                                    lithologies=[dry_sands, wet_soils, silts, clay,
                                                  tills, sandstone, dolomite, limestone, shale, organic_soils],
                                    thick_min=self.NZ,
                                    thick_max=int(Zmax/dh),
                                    deform=deform,
                                    skip_prob=0.0,
                                    ordered=False,
                                    accept_decrease=0.2,
                                    nmax=self.num_layers_max,
                                    halton_seq=self.Halt_seq,
                                    nlayers=self.nlayer)

        # dry soil, saturated soil, altered rock, granite
        sequences = [fine_sediments_seq]
        strati = Stratigraphy(sequences=sequences,Halton_seq=self.Halt_seq)
        self._stratigraphy = strati

        print('self._stratigraphy:', self._stratigraphy)

        properties = self._stratigraphy.properties()
        print('Generated properties:', properties.keys())



        return self._stratigraphy, properties


    def generate_model(self, seed=235):
        props2D, layerids, layers,Halton_reduced = super().generate_model(seed=seed,Halt_seq=self.Halt_seq)
        props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers,Halton_reduced


#Display the summary of the lithologies:
model=ViscoElasticModel()
model.Summary_lithologies(model.stratigraphy, model.properties)
#Generate a model and plot it:
props2d, layerids, layers,Halton_reduced= model.generate_model(seed=235)
model.plot_model(props2d, layers, animated=False, figsize=(16, 8))

#define MyAcquisition class using SeismicAcquisition class from SeismicGenerator.py
class MyAcquisition2(SeismicAcquisition2):
    '''
    We want to make 1.5s long seismograms with 0.02 ms sampling rate.
    If we want a minimum depth investigation of 1m and a maximum depth investigation of 50m, we need to have a grid spacing of = 0.5m.
    We want to have 50m long line of receivers with 96 receivers spaced by 1m.

    We will create a grid of 100m long * 50m depth grid cells with a grid spacing of 0.5m.

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
        self.singleshot = True  # if True, only one shot gather is generated
        self.nab = nab


#plot the aquisition geometry:
aquire=MyAcquisition2()
aquire.plot_acquisition_geometry(path=path_figures)

#props2d['vs'] =np.asfortranarray(props2d['vs'])
#props2d['vp'] =np.asfortranarray(props2d['vp'])
#props2d['rho'] =np.asfortranarray(props2d['rho'])

seismic_acquisition = SeismicGenerator(acquire=aquire)
seismic_acquisition.acquire = aquire
data = seismic_acquisition.compute_data(props=props2d)
# create shotgather using our model and acquisition:
shotgather = ShotGather(model=model, acquire=aquire)
data,_ = shotgather.generate(data)

print('after data generation:')
print('data:',data.shape)
print('data min/max:',data.min(), data.max())
print('data type:',data.dtype)

# apply transforms on data if needed (normalization, standardization ...)

# Normalisation
data = np.array(data)
data = data - np.min(data, axis=0)
data = data / np.max(data, axis=0)

data = torch.tensor(data, dtype=torch.float32)
transform_resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])
data = transform_resize(data)

print('After transforms:')
print('data:',data.shape)
print('data min/max:',data.min(), data.max())
print('data type:',data.dtype)

print(f'VS (shape = {props2d["vs"].shape}):\n',props2d['vs'])
print(f'VP (shape = {props2d["vp"].shape}):\n',props2d['vp'])

#reshape vs and vp to be compatible with deepwave, without the absorbing boundaries:
# absorbing boudaries of nab cells on each side, but not on the top because we want the free surface
print('nab:',nab)
vs_model = props2d["vs"][0:-nab, nab:-nab]
vp_model = props2d["vp"][0:-nab, nab:-nab]



print(f'VS model for deepwave (shape = {vs_model.shape}):\n',vs_model)
print(f'VP model for deepwave (shape = {vp_model.shape}):\n',vp_model)

#plot them:
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(vs_model, cmap='jet', aspect='auto')
plt.colorbar(label='Shear Wave Velocity (m/s)')
plt.title('S Wave Velocity Model (Vs)')
plt.xlabel('Depth (cells)')
plt.ylabel('Distance (cells)')
plt.subplot(1, 2, 2)
plt.imshow(vp_model, cmap='jet', aspect='auto',)
plt.colorbar(label='Primary Wave Velocity (m/s)')
plt.title('P Wave Velocity Model (Vp)')
plt.xlabel('Depth (cells)')
plt.ylabel('Distance (cells)')
plt.tight_layout()
plt.show()
plt.close()

# set parameters for deepwave simulation:
nx = vs_model.shape[1]  # Number of grid points in x direction (328 without absorbing boundaries)
nz = vs_model.shape[0]  # Number of grid points in z direction (100 without absorbing boundaries)
dt_deepwave = dt  # Time step for deepwave simulation
nt = int(1.5/dt_deepwave)       # Number of time steps
f0 = peak_freq                  # Source peak frequency
dh_deepwave = dh                # Grid spacing in meters
pml_width = [0, nab, nab, nab]  # Absorbing frontiers (top,bottom,left,right)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Create velocity models as torch tensors and move to device

vp_model = np.ascontiguousarray(props2d["vp"][0:-nab, nab:-nab])
vs_model = np.ascontiguousarray(props2d["vs"][0:-nab, nab:-nab])
rho_model = np.ascontiguousarray(props2d["rho"][0:-nab, nab:-nab])

vp_tensor = torch.tensor(vp_model, dtype=torch.float32, device=device)
vs_tensor = torch.tensor(vs_model, dtype=torch.float32, device=device)
rho_tensor = torch.tensor(rho_model, dtype=torch.float32, device=device)

#acquisition parameters according to MyAcquisition2 class:
#SOURCE
n_shots = 1              #nombre de tir
n_sources_per_shot = 1  #nombre de source par tir
d_source = 1            #distance entre les sources
src_pos, rec_pos = aquire.set_rec_src()
first_source = src_pos[0]/dh_deepwave - nab #position de la premiere source en cellules
print('first_source:',first_source)
source_depth = 0        #profondeur des sources

#RECEVERS:
n_receivers_per_shot = 96
d_receiver = 1 / dh_deepwave  # spacing between receivers in grid cells (1m / 0.5m = 2 cells)
first_receiver = first_source + d_receiver
receiver_depth = 0


source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long)
source_locations[..., 0] = source_depth
source_locations[:,0,1] =(torch.arange(n_shots) * d_source + first_source)

receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                dtype=torch.long, device=device)
receiver_locations[..., 0] = receiver_depth
receiver_locations[:, :, 1] = (
    (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver)
    .repeat(n_shots, 1)
)

wavelet= seismic_acquisition.seiscl.src_all[:,0]


source_amplitudes0 = (
    (deepwave.wavelets.ricker(peak_freq, nt, dt_deepwave, 1.5 / peak_freq))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)


source_amplitudes = torch.tensor(wavelet,dtype=torch.float32).repeat(n_shots, n_sources_per_shot, 1).to(device)

#plot both source wavelets to compare:
plt.figure(figsize=(10, 5))
plt.plot(source_amplitudes[0,0,:].cpu(), label='SeismicGenerator Wavelet', linestyle='-')
plt.plot(source_amplitudes0[0,0,:].cpu(), label='Deepwave Ricker Wavelet', linestyle='--')
plt.title('Comparison of Source Wavelets')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
plt.close()

# Simulate wave propagation using DeepWave
observed_data = deepwave.elastic(*deepwave.common.vpvsrho_to_lambmubuoyancy(vp_tensor, vs_tensor,
                                               rho_tensor),
        grid_spacing=dh_deepwave,
        dt=dt_deepwave,
        source_amplitudes_y=source_amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        pml_width=pml_width,
        pml_freq=peak_freq,
        )[-2]

print('observed_data shape:', observed_data.shape)
data_img = observed_data.detach().cpu().numpy()[0]

eps = np.finfo(np.float32).eps
trace_rms = np.sqrt(np.sum(data_img**2, axis=0, keepdims=True))
data_img /= trace_rms + eps

panel_max = np.amax(data_img, axis=(0, 1), keepdims=True)
data_img /= panel_max + eps

gain = 1000
data_img = data_img * gain
data_img = np.array(data_img)
data_img = data_img - np.min(data_img)
data_img = data_img / np.max(data_img)

data_resized = transform_resize(data_img)


def plot_shot_gather(data_resized=data_resized,true_image=data,debug=True):
    print('shapes in plot_shot_gather function:')
    print('data_resized shape:', data_resized.shape)
    print('true_image shape:', true_image.shape)
    time_vector = np.linspace(0, 1.5, data_resized.shape[2])
    nb_traces = 96

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(data_resized[0].T, aspect='auto', cmap='gray',
                    extent=[0, nb_traces, time_vector[-1], time_vector[0]])
    axs[0].set_title('Observed Shot Gather')
    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Time (s)')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(true_image[0].cpu(), aspect='auto', cmap='gray',
                    extent=[0, nb_traces, time_vector[-1], time_vector[0]])
    axs[1].set_title('True Shot Gather')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Time (s)')
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.show()


    if debug:
        print('data_resized shape:', data_resized.shape)
        print('true_image shape:', true_image.shape)
        print('mindata_resized:', torch.min(data_resized), 'maxdata_resized:', torch.max(data_resized))
        print('mintrue_image:', torch.min(true_image), 'maxtrue_image:', torch.max(true_image))

plot_shot_gather(data_resized=data_resized,true_image=data,debug=True)


