function data = cornerFinder( nc, filterSize, fileName )
%CORNERS Summary of this function goes here
%   Detailed explanation goes here
    img = imread(fileName);
    if length(size(img)) > 2
        lastIdx = size(img);
        if (lastIdx(3)) == 4
            img(:,:,4) = [];
        end
        img = rgb2gray(img);
    end
    corners = detectHarrisFeatures(img, 'FilterSize', filterSize);
%     [~, valid_corners] = extractFeatures(img, corners);
    imshow(img);hold on;
%     plot(valid_corners);

    points = corners.selectStrongest(nc);
    plot(corners.selectStrongest(nc));
    center = mean(points.Location);
    data = points.Location;
    plot(center(1,1), center(1, 2), 'r*');
end

