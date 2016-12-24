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

function [name, stopTime, trialNumber] = getClipData(xmlDoc)

name = '';
stopTime = 0;

node1 = xmlDoc.getElementsByTagName('trial').item(0);

node2 = node1.getElementsByTagName('number').item(0);
trialNumber = str2double(node2.getTextContent);

nodes = node1.getElementsByTagName('clip');
if (nodes.getLength > 0)
    node2 = nodes.item(0);
    name = char(node2.getTextContent);
end

nodes = node1.getElementsByTagName('stopTime');
if (nodes.getLength > 0)
    node2 = nodes.item(0);
    stopTime = str2double(node2.getTextContent);
end

