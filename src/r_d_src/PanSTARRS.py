import os
import numpy as np
# For resolving objects with tools from MAST
from astroquery.mast import Mast
import mastcasjobs


# For handling ordinary astropy Tables in responses
from astropy.table import Table, vstack
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

# CASJOBS status codes:
# self.status_codes = ("ready", "started", "canceling", "cancelled",
#                      "failed", "finished")
	def upload_obj_idlist(self, jobs, obj_idlist, tblname = 'obj_idlist', chunksz=1000):

		#unique-ify the obj_idlist
		obj_list = list(set(obj_idlist))
		nobj = len(obj_idlist)
		nchunks = int(np.ceil(len(obj_list)/chunksz))

		jobs.drop_table_if_exists(tblname)

		#upload in chunks
		for chunk in range(nchunks):
			if chunk == 0:
				hdr = 'objID,\n'
				exists=False
			else:
				hdr = ''
				exists=True

			# make phony csv string
			idlist =  obj_list[ chunk*chunksz : (chunk+1)*chunksz ]
			objcsv = hdr + ',\n'.join([str(id) for id in idlist])

			jobs.upload_table(tblname, objcsv, exists=exists)

		#how many uploaded?
		qry = f'select count(*) as RecCount from MyDB.{tblname}'
		res = jobs.quick(qry, task_name="upload verify")
		reccount = res['RecCount'][0]
		print(f'Records in Table: {reccount}')

	def from_obj_idlist(self,  obj_idlist, columns=None, tblname='obj_idlist',
		     user=None, pwd=None, #values from environment variables if not spec'd
		     verbose=False):
		if columns is not None:
			cols=columns
		else:
			cols = PanSTARRS1.ps1_collist

		user = os.environ.get('CASJOBS_USERID') if user is None else user
		pwd = os.environ.get('CASJOBS_PW') if pwd is None else pwd

		# casjob states
		active_states = ['ready', 'started', 'canceling']
		finish_states = ['cancelled', 'failed', 'finished']

		#start the CASJOBS
		jobs = mastcasjobs.MastCasJobs(username=user, password=pwd, request_type='POST',context="PanSTARRS_DR2")
		#upload the obj_idlist
		self.upload_obj_idlist(jobs, obj_idlist, tblname=tblname)

		#make the column list
		sqlcols = ','.join(['mo.'+c for c in cols])

		#build the query
		qry = f'Select upl.objID as Req_objID,'+sqlcols+f' from MyDB.{tblname} upl' \
				' into MyDB.obj_query' \
				' left join dbo.MeanObjectView mo on upl.objID = mo.objID'

		#ditch the output table
		jobs.drop_table_if_exists("obj_query")
		
		#run the query
		job_id = jobs.submit(qry, context='PanSTARRS_DR2')
		
		#hang out till the job finishes
		while jobs.monitor(job_id)[1] in active_states:
			continue

		tbl = jobs.get_table('MyDB.obj_query')

		# mask up the pstars result
		tbl = Table(tbl, masked=True, copy=False)
		cols = [col  for col in tbl.columns if col not in ['Req_objID']]
		for col in cols:
			tbl[col].mask = np.logical_or(tbl[col].mask, tbl[col] == -999)

		nrows = len(tbl)
		nvalid = nrows - tbl['objID'].mask.sum()

		print(f'Rows requested: {nrows}, valid rows returned: {nvalid}')

		tbl.add_index('Req_objID')
		self.objs = tbl


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
		    135990576794699063, 136000579888507545, 136290579648226866, 133700582422369377, 123, 456, 789]
    
    ps1 = PanSTARRS1(name='Test Retrieval')
    ps1.from_obj_idlist(idlist)

    print(ps1.objs)

