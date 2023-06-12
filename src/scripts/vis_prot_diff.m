subplot(2,2,1);
imshow(squeeze(origPrototypes(:,:,1,1)));
subplot(2,2,2);
imshow(squeeze(origPrototypes(:,:,1,2)));

subplot(2, 2, 3);
protdiff = result.averageRun.prototypes(1,:) - result.averageRun.prototypes(2,:);
protdiff_dec = autoenc.decode(protdiff);
protdiff2 = result.averageRun.prototypes(2,:) - result.averageRun.prototypes(1,:);
protdiff_dec2 = autoenc.decode(protdiff2);
imshow(protdiff_dec);
subplot(2,2,4);
imshow(protdiff_dec2);