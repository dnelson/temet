import readsubfHDF5 
import snapHDF5 
import numpy as np
import os
import sys

FlagRead, cat,GroupOffset, HaloOffset, multiple, filename, Parttype, FileTypeNumbers, FileNum = False, None, None, None, None, None, [], None, None

def reset():
        global FlagRead, cat, GroupOffset, HaloOffset, multiple, filename, Parttype, FileTypeNumbers, FileNum
        FlagRead, cat, GroupOffset, HaloOffset, multiple, filename, Parttype, FileTypeNumbers, FileNum = False, None, None, None, None, None, [], None, None



def readhalo(base, snapbase, num, block_name, parttype, fof_num, sub_num, long_ids=False, double_output=False, verbose=False):
	global FlagRead, cat, GroupOffset, HaloOffset, multiple, filename, Parttype, FileTypeNumbers, FileNum


	if (FlagRead==False) | ((parttype in Parttype) == False):	
		if (verbose):
			print "READHALO: INITIAL READ"

		#add parttype to list
		Parttype.append(parttype)

		if (verbose):
			print "READHALO: Parttype = ", Parttype

		#read in catalog
		cat = readsubfHDF5.subfind_catalog(base, num, long_ids=long_ids, double_output=double_output, keysel=["GroupLenType","GroupNsubs","GroupFirstSub","SubhaloLenType","SubhaloMassType"])

		if (cat.ngroups==0):
			if (verbose):
				print "READHALO: no groups in catalog... returning"
			return

		
		if (FlagRead==False):
			GroupOffset = np.zeros([cat.ngroups, 6], dtype="int64")
			HaloOffset = np.zeros([cat.nsubs, 6], dtype="int64")
	
	                filename = base+"/"+snapbase+"_"+str(num).zfill(3)
        	        multiple=False
                	if (os.path.exists(filename+".hdf5")==False):
                        	filename = base+"/snapdir_"+str(num).zfill(3)+"/"+snapbase+"_"+str(num).zfill(3)+"."+str(0)
	                        multiple=True
        	        if (os.path.exists(filename+".hdf5")==False):
                	        print "READHALO: [error] file not found : ", filename
                        	sys.exit()

			FlagRead=True


	        #construct offset tables
		k=0
		for i in range(0, cat.ngroups):
			if (i>0):
				GroupOffset[i, parttype] =  GroupOffset[i-1, parttype] + cat.GroupLenType[i-1, parttype]
			if (cat.GroupNsubs[i]>0):
				HaloOffset[k, parttype] = GroupOffset[i, parttype]
				k+=1
				for j in range(1, cat.GroupNsubs[i]):
		       		        HaloOffset[k, parttype] =  HaloOffset[k-1, parttype] + cat.SubhaloLenType[k-1, parttype]
					k+=1
		if (k!=cat.nsubs):
			print "READHALO: problem with offset table", k, cat.nsubs
			sys.exit()

		#construct file tables
		if (multiple):
			filename = base+"/snapdir_"+str(num).zfill(3)+"/"+snapbase+"_"+str(num).zfill(3)+"."+str(0)
		else:
			filename = base+"/"+snapbase+"_"+str(num).zfill(3)

		head = snapHDF5.snapshot_header(filename)
		FileNum = head.filenum

		FileTypeNumbers = np.zeros([FileNum, 6], dtype="int64") 
		cumcount = np.zeros(6, dtype="int64")

		for fnr in range(0, FileNum-1):
			if (multiple):
				filename = base+"/snapdir_"+str(num).zfill(3)+"/"+snapbase+"_"+str(num).zfill(3)+"."+str(fnr)
			else:
				filename = base+"/"+snapbase+"_"+str(num).zfill(3)

			if (verbose):
				print "READHALO: initial reading file :", filename

			head = snapHDF5.snapshot_header(filename)
	
			cumcount[:] += head.npart[:]
			FileTypeNumbers[fnr+1, :] = cumcount[:]
			
	
	if (sub_num>=0) & (fof_num < 0):
		off = HaloOffset[sub_num, parttype]
        	left = cat.SubhaloLenType[sub_num, parttype]
		if (verbose):
			print "READHALO: nr / particle # / mass :", sub_num, cat.SubhaloLenType[sub_num, parttype], cat.SubhaloMassType[sub_num, parttype].astype("float64")
	if (fof_num>=0) & (sub_num < 0):
                off = GroupOffset[fof_num, parttype]
                left = cat.GroupLenType[fof_num, parttype]
		if (verbose):
			print "READHALO: nr / particle # / mass :", fof_num, cat.GroupLenType[fof_num, parttype], cat.GroupMassType[fof_num, parttype].astype("float64")
	if (sub_num>=0) & (fof_num>=0):
		real_sub_num = sub_num + cat.GroupFirstSub[fof_num]
		off = HaloOffset[real_sub_num, parttype]
		left = cat.SubhaloLenType[real_sub_num, parttype]
                if (verbose):
                        print "READHALO: nr / particle # / mass :", real_sub_num, cat.SubhaloLenType[real_sub_num, parttype], cat.SubhaloMassType[real_sub_num, parttype].astype("float64")
		

        if (left==0):
		if (verbose):
	                print "READHALO: no particles of type... returning"
                return


	#get first file that contains particles of required halo/fof/etc
	print "Offset: " + str(off)
	print "FileOffsetsType: " + ' '.join( [str(x) for x in FileTypeNumbers[:,parttype]] )
    
	findex = np.argmax(FileTypeNumbers[:, parttype] > off) - 1
	#in case we reached the end argmax returns 0
	if (findex == -1): 
		findex = FileNum - 1

	if (verbose):
		print "READHALO: first file that contains particles =", findex

	for fnr in range(0, findex):
		off -= FileTypeNumbers[fnr+1, parttype] - FileTypeNumbers[fnr, parttype] 

	#read data from file
	first=True
	for fnr in range(findex, FileNum):
		if (multiple):
	                filename = base+"/snapdir_"+str(num).zfill(3)+"/"+snapbase+"_"+str(num).zfill(3)+"."+str(fnr)
		else:
                        filename = base+"/"+snapbase+"_"+str(num).zfill(3)
	
		if (verbose):
			print "READHALO: reading file :", filename
	
		head = snapHDF5.snapshot_header(filename)
		nloc = head.npart[parttype]

                if (nloc > off):
			if (verbose):
				print "READHALO: data"
	                start = off
        	        if (nloc - off > left):
	                	count = left	
			else:
                    		count = nloc - off

			if (first==True):	
				data = snapHDF5.read_block(filename, block_name, parttype, slab_start=start, slab_len=count)
				first=False
			else:
				data = np.append(data, snapHDF5.read_block(filename, block_name, parttype, slab_start=start, slab_len=count), axis=0)

        	        left -= count
        	        off += count
		if (left==0):
			break
		off -= nloc

	return data

def testRead():
    #read specific subhalo/group particles
    snapnum=135
    base="/n/ghernquist/Illustris/Runs/Illustris-1/"
    type=4
    #grpnr=50975
    #subnr=0 # subhalo number 692283
    grpnr = -1
    subnr = 692283
    reset()
    pos=readhalo(base, "snap", snapnum, "POS ", type, grpnr, subnr, long_ids=True, double_output=False, verbose=True)
    return pos