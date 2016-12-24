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

function grid = getLetterGrid(xmlDoc)

node1 = xmlDoc.getElementsByTagName('letterGrid').item(0);
node2 = node1.getElementsByTagName('width').item(0);
grid.width = str2double(node2.getTextContent);
node2 = node1.getElementsByTagName('height').item(0);
grid.height = str2double(node2.getTextContent);

nodes = node1.getElementsByTagName('displayTime');
grid.time = 1000;
if (nodes.getLength > 0)
    node2 = nodes.item(0);
    grid.time = str2double(node2.getTextContent);
end

nodes = node1.getElementsByTagName('density');
grid.density = 0;
if (nodes.getLength > 0)
    node2 = nodes.item(0);
    grid.density = str2double(node2.getTextContent);
end

nodes = node1.getElementsByTagName('letter');

nl = nodes.getLength;
grid.label = cell(nl, 1);
grid.location = zeros(nl, 2);

for i = 0 : nl-1
    letterNode = nodes.item(i);
    
    nd = letterNode.getElementsByTagName('label').item(0);
    grid.label{i+1} = char(nd.getTextContent);
    
    nd = letterNode.getElementsByTagName('x').item(0);
    grid.location(i+1, 1) = str2double(nd.getTextContent);
    nd = letterNode.getElementsByTagName('y').item(0);
    grid.location(i+1, 2) = str2double(nd.getTextContent);
end

