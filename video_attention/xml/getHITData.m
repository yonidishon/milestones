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

function [hid, wid, aid] = getHITData(xmlDoc)

node1 = xmlDoc.getElementsByTagName('HITData').item(0);

node2 = node1.getElementsByTagName('HITID').item(0);
hid = char(node2.getTextContent);
node2 = node1.getElementsByTagName('workerID').item(0);
wid = char(node2.getTextContent);
node2 = node1.getElementsByTagName('assignmentID').item(0);
aid = char(node2.getTextContent);

