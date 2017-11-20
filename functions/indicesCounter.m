function countedIndices = indicesCounter(fisherCell,nFeatures,totalFeatures)
countedIndices = zeros(1,totalFeatures);
for indInner = 1:size(fisherCell,1)
    for indOuter = 1:size(fisherCell,2)
        for indNFeatures=1:nFeatures
            countedIndices(fisherCell{indInner,indOuter}(indNFeatures)) =...
               countedIndices(fisherCell{indInner,indOuter}(indNFeatures)) + 1;
        end
    end
end