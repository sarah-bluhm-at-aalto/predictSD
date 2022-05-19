paths = getArgument();
argList = split(paths, ";;");
outPath = argList[0];
imagePath = argList[1];
labelPath = argList[2];
chNum = parseInt(argList[3]) + 1;
lutName = argList[4];

// Flatten label image based on maximum intensity, i.e. largest label ID is drawn on top of smaller
open(labelPath);
lblName=getTitle();
if(nSlices > 1) {
    run("Z Project...", "projection=[Max Intensity]");
}
run(lutName);
pLabel=getTitle();
close(lblName);

// Flatten microscopy image based on max intensity
open(imagePath);
imgName=getTitle();
if(nSlices > 1) {
    run("Z Project...", "projection=[Max Intensity]");
}
run("Duplicate...", "duplicate channels="+chNum+"");
close(imgName);
run("Grays");
run("Enhance Contrast", "saturated=0.35");

// Overlay
run("Add Image...", "image="+pLabel+" x=0 y=0 opacity=70 zero");
save(outPath);
run("Close All");
//run("Quit");