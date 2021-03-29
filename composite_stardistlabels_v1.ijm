//1st tiffs folder. REPLACE BACKSLASHES
Path1 = '//ad.helsinki.fi/home/j/jgeorge/Desktop/1_first_tiffs'
// 2nd tiffs folder
Path2 = '//ad.helsinki.fi/home/j/jgeorge/Desktop/2_second_tiffs'
// Write save path you want to save final tiffs in.
Savpath = '//ad.helsinki.fi/home/j/jgeorge/Desktop/3_saved'

list = getFileList(Path1);
//print(Path1);

for (i = 0; i <list.length; i++) {
	//print(i);
	if(endsWith(list[i], ".tiff"));
	path = Path1 +"/"+ list[i];
	//print(path);
	open(path);
	FileTitle1=getTitle();
	FileTitle1ext = replace(FileTitle1, ".tiff", ""); 
	
	P2tiff = Path2 + "/" + FileTitle1ext + "_Ch=0.labels.tif";
	open(P2tiff);
	FileTitle2=getTitle();
	
	run("Z Project...", "projection=[Average Intensity]");
	
	selectWindow(FileTitle1);
	run("Duplicate...", "duplicate channels=1");
	//Keep GFP and z project
	run("Z Project...", "projection=[Average Intensity]");
	zPGFP=getTitle();
	selectWindow(FileTitle2);
	run("Z Project...", "projection=[Average Intensity]");
	zPLABEL=getTitle();
	run("Merge Channels...", "c1="+zPGFP+" c2="+zPLABEL+" create");
	Stack.setDisplayMode("color");
	Stack.setChannel(1);
	run("Green");
	Stack.setChannel(2);
	run("Cyan");
	Stack.setDisplayMode("composite");
	savnam = Savpath + "/" + FileTitle1;
	//print(savnam);
	save(savnam);
	run("Close All");
	}