function filtcands = xxx_filtcands(cands, allowedtypes)
 %filter candidate to only allowedtypes
for ii =1:length(cands)
    filt_idx = zeros(length(cands{ii}),1);
    for jj = 1:length(cands{ii})
        if ~ismember(cands{ii}{jj}.type,allowedtypes) 
            filt_idx(jj)=1;
        end 
    end
    cands{ii}(find(filt_idx))=[];
end
filtcands = cands;