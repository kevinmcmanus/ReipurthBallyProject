from astropy.samp import SAMPIntegratedClient
frameid = 'SUPA01564820'

client= SAMPIntegratedClient()
client.connect()
client_list = client.get_registered_clients()
print(f'Registered clients: {client_list}')

for cl in client_list:
    print(f'Client: {cl}')
    print(client.get_metadata(cl))

r=client.ecall_and_wait('c1', 'ds9.set', '10', cmd='scale log')
print(r)
r=client.ecall_and_wait('c1', 'ds9.set', '10', cmd= 'scale mode 99.5')
print(r)

params = {}
params['url'] = r'file:///home/kevin/Documents/M8/N-A-L656/calibrated/' + frameid+'.fits'
params['name'] = frameid

message = {}
message["samp.mtype"] = "image.load.fits"
message["samp.params"] = params

client.notify_all(message)

r=client.ecall_and_wait('c1', 'ds9.set', '10', cmd='zoom to fit')
#r=client.ecall_and_wait("c1","ds9.set","10",cmd="zoom to fit")
print(r)

# oject catalog
params = {}
params['url'] = r'file:///home/kevin/Documents/M8/N-A-L656/objcat/' + frameid+'.xml'
params['name'] = frameid + ' Object Catalog'
message = {}
message["samp.mtype"] = "table.load.votable"
message["samp.params"] = params

client.notify_all(message)
r=client.ecall_and_wait('c1', 'ds9.set', '10', cmd= 'catalog symbol text $objid')
print(r)
# gaia catalog
params = {}
params['url'] = r'file:///home/kevin/Documents/M8/N-A-L656/gaiacat/' + frameid+'.xml'
params['name'] = frameid + ' Gaia Catalog'
message = {}
message["samp.mtype"] = "table.load.votable"
message["samp.params"] = params

client.notify_all(message)
r=client.ecall_and_wait('c1', 'ds9.set', '10', cmd='catalog symbol load /home/kevin/Documents/ds9/sym/gaia_x.sym')
client.ecall_and_wait('c1', 'ds9.set', '10', cmd='catalog ra ra_obsdate' )
client.ecall_and_wait('c1', 'ds9.set', '10', cmd='catalog dec dec_obsdate' )

# get the match vectors (region file)
regname = '/home/kevin/Documents/M8/N-A-L656/matchregion/'+ frameid + '_init.reg'
client.ecall_and_wait('c1', 'ds9.set', '10', cmd = 'region load '+regname)

client.disconnect()




# from astropy.samp import SAMPIntegratedClient

# ds9 = SAMPIntegratedClient()

# ds9.connect()

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="rgb")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="rgb red")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="url http://ds9.si.edu/download/data/673nmos.fits")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="zscale")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="rgb green")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="url http://ds9.si.edu/download/data/656nmos.fits")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="zscale")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="rgb blue")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="url http://ds9.si.edu/download/data/502nmos.fits")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="zscale")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="rotate 270")

# ds9.ecall_and_wait("c1","ds9.set","10",cmd="zoom to fit")

# ds9.disconnect()