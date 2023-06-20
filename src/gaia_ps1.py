import tempfile, os
from astroquery.gaia import Gaia

def source_id_to_xmlfile(source_idlist, sidcol='typed_id', table_id='source_idlist'):
	#need tempfile for source id list
	fh =  tempfile.mkstemp()
	os.close(fh[0]) #fh[0] is the file descriptor; fh[1] is the path

	#xml-ify the source_idlist to a file
	tbl = Table({sidcol:source_idlist})
	tbl.write(fh[1], table_id=table_id, format='votable', overwrite=True)

	return table_id, fh[1], sidcol

from astropy.table import Table

def gaiadr3toPanStarrs1(source_idlist,nearest=True):
	"""
	returns the panstarrs1 cross matches for the given source_idlist
	"""

	upload_tablename, upload_resource, sidcol = source_id_to_xmlfile(source_idlist)

	query_str = ' '.join([f'SELECT tu.{sidcol}, ps1.*',
				f'from tap_upload.{upload_tablename} tu left join gaiadr3.panstarrs1_best_neighbour ps1',
				f'on tu.{sidcol} = ps1.source_id'])
	try:
		job = Gaia.launch_job_async(query=query_str,
								upload_resource=upload_resource,
								upload_table_name=upload_tablename)
		
		# df = job.get_results().to_pandas()
		tbl = job.get_results()
	finally:
		os.remove(upload_resource)
	
	# if nearest:
	# 	#just return the nearest dr3 source id based on angular distance
	# 	ret_df = df.sort_values(['dr2_source_id','angular_distance']).groupby('dr2_source_id',
	# 				as_index=False).first().set_index(sidcol)
	# else:
	# 	ret_df = df.set_index(sidcol)

	# return ret_df
	tbl.add_index('typed_id')
	return tbl

if __name__ == '__main__':
	idlist = [63449521800719488,
				63502259702709888,
				63507860340462208,
				63527827643105408,
				65588484235454592,
				65589137070475264,
				65603946117575936,
				65605354866829184,
				63547270960574336,
				63547378335443200]
	
	tbl = gaiadr3toPanStarrs1(idlist)
	print(tbl)