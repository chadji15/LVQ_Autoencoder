numprot = size(origPrototypes,4);
for i=1:numprot
subplot(2,numprot,i);
imshow(squeeze(origPrototypes(:,:,1,i)),[]);
end

subplot(2, numprot, 3);
protdiff = squeeze(origPrototypes(:,:,1,1) - origPrototypes(:,:,1,2));
protdiff2 = squeeze(origPrototypes(:,:,1,2) - origPrototypes(:,:,1,1));
imshow(protdiff,[]);
subplot(2,numprot,4);
imshow(protdiff2,[]);

