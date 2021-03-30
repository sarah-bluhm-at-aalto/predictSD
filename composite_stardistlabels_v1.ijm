// N.B. Make sure paths contain FORWARDSLASHES and not BACKSLASHES

//1st tiffs folder. 
Path1 = 'E:/Duuni/StarDist_Annotation/newnewImg/omat'

// 2nd tiffs folder
Path2 = 'E:/Duuni/StarDist_Annotation/newnewImg/omat/masks'

// Write save path you want to save final tiffs in.
Savpath = 'E:/Duuni/StarDist_Annotation/newnewImg/omat/comp'

// Give label file extension, e.g. image's name + the extension: "_Ch=0.labels.tif"
labelExt = '.tif'

list = getFileList(Path1);
//print(Path1);

for (i = 0; i <list.length; i++) {
	print(list[i]);
	if(endsWith(list[i], ".tif") | endsWith(list[i], ".tiff")) {
		path = Path1 +"/"+ list[i];
		//print(path);
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
		run("Duplicate...", "duplicate channels=1");
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
		savnam = Savpath + "/" + FileTitle1;
		//print(savnam);
		save(savnam);
		run("Close All");
	}
}