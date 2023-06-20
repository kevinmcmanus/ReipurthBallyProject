# Use the pyvo library as our client to the data service.
import pyvo as vo

# For resolving objects with tools from MAST
from astroquery.mast import Mast

# For handling ordinary astropy Tables in responses
from astropy.table import Table
from matplotlib import pyplot as plt

class PanSTARRS1():
	ps1_collist = ['objID', 'RAMean', 'DecMean', 
				   'nDetections', 'ng', 'nr', 'ni', 'nz', 'ny',
					'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag',
					'zMeanPSFMag', 'yMeanPSFMag']
	def __init__(self, **kwargs):
		self.name=kwargs.pop('name',None)
		self.description = kwargs.pop('description', None)

		self.coords = None
		self.tap_query_string = None
		self.objs = None

	def from_source_id_list(self, source_idlist, columns=None):
		if columns is not None:
			cols=columns
		else:
			cols = PanSTARRS1.ps1_collist

		#make the query string
		sqlcols = ','.join(cols)
		idlist = ','.join([str(id)for id in source_idlist])
		qrystr = 'SELECT '+sqlcols+' from dbo.MeanObjectView where objID in ('+idlist+')'

		#set up the tap service
		TAP_service = vo.dal.TAPService("https://vao.stsci.edu/PS1DR2/tapservice.aspx")
		#let 'er rip
		job = TAP_service.run_async(qrystr)

		self.objs = job.to_table()
		self.objs.add_index('objID')
		print(f'Query returned {len(self.objs)} records')

	def get_colors(self):

		#color
		R_Z = self.objs['rMeanPSFMag']- self.objs['zMeanPSFMag']
		# misuse of upper case G; we're acutally returning gmag (apparent mag)
		G = self.objs['gMeanPSFMag']
		return R_Z, G
	
	
	def plot_hrdiagram(self, **kwargs):
		ax = kwargs.pop('ax',None)
		title = kwargs.pop('title', 'HR Diagram')
		label = kwargs.pop('label',self.name)
		s = kwargs.pop('s', 1) #default size = 1
		xlim = kwargs.pop('xlim', (-1,5))
		ylim = kwargs.pop('ylim', (-5, 25))

   
		if ax is None:
			yax = plt.subplot(111)
		else:
			yax = ax

		R_Z, G = self.get_colors()

		pcm = yax.scatter(R_Z, G, label=label, s=s, **kwargs)

		yax.set_xlim(xlim)
		yax.set_ylim(ylim)
		if not yax.yaxis_inverted():
			yax.invert_yaxis()


		yax.set_title(title)
		yax.set_ylabel(r'$gMag$')
		yax.set_xlabel(r'$rMag - zMag$')
		if ax is None:
			yax.legend()
			
		return pcm


if __name__ == "__main__":
	
    idlist = [133230557739126875, 133300565929697162, 133610565324584819, 135360582688022104,
            135440583154615177, 135570582568226479, 135660582923743941, 133530560026408711, 135870586557031252,
	        136070588787927387, 133880562203025858, 134020565669993767, 136010585542907346, 133270578192632103,
		    135790578312136884, 135930578159509145, 133160574599380501, 135950579631614104, 135990577144305170,
		    135990576794699063, 136000579888507545, 136290579648226866, 133700582422369377]
    
    ps1 = PanSTARRS1(name='Test Retrieval')
    ps1.from_source_id_list(idlist)

    print(ps1.objs)

