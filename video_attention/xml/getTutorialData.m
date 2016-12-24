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

function [isTutorial, loc] = getTutorialData(xmlDoc)

loc = [nan nan];
isTutorial = true;

node1 = xmlDoc.getElementsByTagName('tutorial').item(0);
nodes = node1.getElementsByTagName('active');
if (nodes.getLength > 0)
    node2 = nodes.item(0);
    str = node2.getTextContent;
    if (strcmp(str, 'true'))
        isTutorial = true;
    else
        isTutorial = false;
    end
end

if (isTutorial)
    node2 = node1.getElementsByTagName('ballX').item(0);
    node3 = node1.getElementsByTagName('ballY').item(0);
    loc = [str2double(node2.getTextContent), str2double(node3.getTextContent)];
end

