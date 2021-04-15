paths = getArgument();
argList = split(paths, ";;");

chn = parseInt(argList[3]) + 1;

// Flatten microscopy image based on average intensity
open(argList[1]);
run("Duplicate...", "duplicate channels="+chn+"");
run("Enhance Contrast", "saturated=0.35");
run("Z Project...", "projection=[Average Intensity]");
pImg=getTitle();

// Flatten label image based on maximum intensity, i.e. largest label ID is drawn on top of smaller
open(argList[2]);
run("Z Project...", "projection=[Max Intensity]");
pLabel=getTitle();

// Merge flattened images and set color maps
run("Merge Channels...", "c1=["+pImg+"] c2=["+pLabel+"] create");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Grays");
Stack.setChannel(2);
run("glasbey_inverted");
Stack.setDisplayMode("composite");

save(argList[0]);
run("Close All");