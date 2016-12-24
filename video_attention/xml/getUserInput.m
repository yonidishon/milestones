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

function [lbl, loc, error, valid] = getUserInput(xmlDoc)

loc = [nan nan];
valid = false;
error = inf;

% get the input letter
node1 = xmlDoc.getElementsByTagName('input').item(0);
node2 = node1.getElementsByTagName('letter').item(0);
node3 = node2.getElementsByTagName('label').item(0);
lbl = char(node3.getTextContent);

node4 = node1.getElementsByTagName('valid');
if (node4.getLength > 0)
    node5 = node4.item(0);
    str = char(node5.getTextContent);
    if (strcmp(str, 'true'))
        valid = true;
        
        node2 = node1.getElementsByTagName('location').item(0);
        node3 = node2.getElementsByTagName('x').item(0);
        node4 = node2.getElementsByTagName('y').item(0);
        loc = [str2double(node3.getTextContent), str2double(node4.getTextContent)];
        
        node2 = node1.getElementsByTagName('error').item(0);
        error = str2double(node2.getTextContent);
    else
        valid = false;
    end
end

