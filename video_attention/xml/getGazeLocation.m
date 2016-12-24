% ADOBE PROPRIETARY INFORMATION
% 
% Use is governed by the license in the attached LICENSE.TXT file
% 
% 
% Copyright 2011 Adobe Systems Incorporated
% All Rights Reserved.
% 
% NOTICE:  All information contained herein is, and remains
% the property of Adobe Systems Incorporated and its suppliers,
% if any.  The intellectual and technical concepts contained
% herein are proprietary to Adobe Systems Incorporated and its
% suppliers and may be covered by U.S. and Foreign Patents,
% patents in process, and are protected by trade secret or copyright law.
% Dissemination of this information or reproduction of this material
% is strictly forbidden unless prior written permission is obtained
% from Adobe Systems Incorporated.
% 

function [loc, err, found, lbl] = getGazeLocation(xmlDoc, letterGrid)

loc = [-1 -1];

% get the input letter
node1 = xmlDoc.getElementsByTagName('input').item(0);
node2 = node1.getElementsByTagName('letter').item(0);
node3 = node2.getElementsByTagName('label').item(0);
lbl = node3.getTextContent;

% look for the EXACT label on the grid
found = false;
for i = 1:length(letterGrid.label)
    if (strcmp(lbl, letterGrid.label{i}))
        loc = letterGrid.location(i, :);
        found = true;
        break;
    end
end

if (~found)
    % look for lowercase
    for i = 1:length(letterGrid.label)
        if (strcmpi(lbl, letterGrid.label{i}))
            loc = letterGrid.location(i, :);
            found = true;
            break;
        end
    end
end

% TODO error is hardcoded
err = [60 60];

