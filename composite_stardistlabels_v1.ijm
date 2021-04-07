// N.B. Make sure paths contain FORWARDSLASHES and not BACKSLASHES

//1st tiffs folder. Original tiffs. 
Path1 = 'G:/010421/PROCESSED_TIFFS'

//2nd tiffs folder (labels).
Path2 = 'G:/010421/labels'

// Write save path you want to save final tiffs in.
Savpath = 'G:/010421/Label overlays/GFP'

// Add the label channel number you want to overlay e.g. Ch0 = 0 (usually GFP), Ch1 = 1 (Usually DAPI)
labChannel = 0

// Label file extension, e.g. image's name + the extension: "_Ch=0.labels.tif"
labelExt = '_Ch='+labChannel+'.labels.tif'
print(labelExt)

// Creates the number of the channel from the original tiff which will be duplicated below
dupNumber = labChannel + 1

list = getFileList(Path1);


setBatchMode(true);
for (i = 0; i <list.length; i++) {
	print(list[i]);
	if(endsWith(list[i], ".tif") | endsWith(list[i], ".tiff")) {
		path = Path1 +"/"+ list[i];
		open(path);
		FileTitle1=getTitle();

		if(endsWith(list[i], ".tiff")) {
		    FileTitle1ext = replace(FileTitle1, ".tiff", "");
		} else {
		    FileTitle1ext = replace(FileTitle1, ".tif", "");
		}
		
		P2tiff = Path2 + "/" + FileTitle1ext + labelExt;
		open(P2tiff);
		run("Enhance Contrast", "saturated=0.35");
		FileTitle2=getTitle();
		run("Z Project...", "projection=[Average Intensity]");
		
		selectWindow(FileTitle1);
		// Change if you want e.g DAPI (duplicate channels=2) or GFP (Duplicate channels=1)
		run("Duplicate...", "duplicate channels="+dupNumber+"");
		//Keep single channel (GFP or DAPI) and z project
		run("Z Project...", "projection=[Average Intensity]");
		zPGFP=getTitle();
		selectWindow(FileTitle2);
		run("Z Project...", "projection=[Max Intensity]");
		zPLABEL=getTitle();
		run("Merge Channels...", "c1="+zPGFP+" c2="+zPLABEL+" create");
		Stack.setDisplayMode("color");
		Stack.setChannel(1);
		run("Grays");
		Stack.setChannel(2);
		run("glasbey_inverted");
		Stack.setDisplayMode("composite");
		savnam = Savpath + "/" + FileTitle1ext + '_Ch'+labChannel+'_overlay';
		//print(savnam);
		save(savnam);
		close("*");
	}
}