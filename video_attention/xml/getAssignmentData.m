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

function [browserName, browserSize, screenSize] = getAssignmentData(xmlDoc)

browserSize = [0 0];
screenSize = [0 0];
browserName = '';

nodes = xmlDoc.getElementsByTagName('userData');

if (nodes.getLength > 0)
    node1 = nodes.item(0);
    node2 = node1.getElementsByTagName('screen').item(0);
    
    node3 = node2.getElementsByTagName('width').item(0);
    screenSize(1) = str2double(node3.getTextContent);
    node3 = node2.getElementsByTagName('height').item(0);
    screenSize(2) = str2double(node3.getTextContent);
    
    node2 = node1.getElementsByTagName('browser').item(0);
    
    node3 = node2.getElementsByTagName('width').item(0);
    browserSize(1) = str2double(node3.getTextContent);
    node3 = node2.getElementsByTagName('height').item(0);
    browserSize(2) = str2double(node3.getTextContent);
    node3 = node2.getElementsByTagName('name').item(0);
    browserName = char(node3.getTextContent);
end

