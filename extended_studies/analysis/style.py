import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7','#fcfcfb'
GREY = '#a3a29c'
SEQ = LinearSegmentedColormap.from_list('seqblue', ['#f3f8fe','#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#184f95','#3987e5','#9ec5f4','#f0efec','#f2a3a2','#e34948','#9e2424'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
  'xtick.color':MUTED,'ytick.color':MUTED,'axes.titlecolor':INK,'axes.titlesize':10.5,'axes.titleweight':'bold',
  'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'axes.spines.top':False,'axes.spines.right':False,
  'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'legend.frameon':False,'lines.linewidth':2,
  'lines.markersize':6,'figure.dpi':110,'savefig.dpi':160,'savefig.bbox':'tight'})
TIERS = ['A_smooth','B_gaussian_bump','C_inclusion','D_layered','E_grf','F_defects']
TNAME = {'A_smooth':'A smooth','B_gaussian_bump':'B bump','C_inclusion':'C inclusion','D_layered':'D layered','E_grf':'E random field','F_defects':'F defects'}
def note(fig, s): fig.text(0.01, -0.02, s, ha='left', va='top', fontsize=8, color=INK2, wrap=True)
