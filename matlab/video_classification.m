v = VideoReader('gta5_pedestrian.mp4');
fps = 0;
out = VideoWriter('result.avi');
open(out);
while hasFrame(v)
    % Read frames from video
    im = readFrame(v);
    im = imresize(im, [480, 640]);
    
    % Call MEX function for pednet prediction
    tic;
    ped_bboxes = pedDetect_predict_mex(im);
    newt = toc;
    
    % fps
    fps = .9 * fps + .1 * (1/newt);
    
    % display
    outputImage = insertShape(im, 'Rectangle', ped_bboxes, 'LineWidth', 3);
    writeImage = imresize(outputImage, [1080, 1920]);
    writeVideo(out, writeImage);
    imshow(outputImage)
    pause(0.2)
end
close(out);