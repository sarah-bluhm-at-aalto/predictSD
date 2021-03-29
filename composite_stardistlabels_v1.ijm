//1st tiffs folder. REPLACE BACKSLASHES
Path1 = '//ad.helsinki.fi/home/j/jgeorge/Desktop/1_first_tiffs'
// 2nd tiffs folder
Path2 = '//ad.helsinki.fi/home/j/jgeorge/Desktop/2_second_tiffs'
// Write save path you want to save final tiffs in.
Savpath = '//ad.helsinki.fi/home/j/jgeorge/Desktop/3_saved'

list = getFileList(Path1);
//print(Path1);

for (i = 0; i <list.length; i++) {
	print(list[i]);
	if(endsWith(list[i], ".tif") | endsWith(list[i], ".tiff")) {
		path = Path1 +"/"+ list[i];
		//print(path);
		open(path);
		FileTitle1=getTitle();
		FileTitle1ext = replace(FileTitle1, ".tiff", "");
		FileTitle1ext = replace(FileTitle1, ".tif", "");
		
		P2tiff = Path2 + "/" + FileTitle1ext + labelExt;
		open(P2tiff);
		run("Enhance Contrast", "saturated=0.35");
		FileTitle2=getTitle();
		
		run("Z Project...", "projection=[Average Intensity]");
		
		selectWindow(FileTitle1);
		run("Duplicate...", "duplicate channels=1");
		//Keep GFP and z project
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