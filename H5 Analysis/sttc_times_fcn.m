% alex dunn, cambridge, 2024
% create adjacency matrix using sttc based on spike times
% based on previous script which used spike matrix as input, parallel
% computing option needs updating
% for data formatted as spike times and a list of channel/neuron numbers
% corresponding to each spike time, this is quicker and uses less memory
% than creating a spike matrix first meaning it can be used for longer
% recordings / larger datasets and array sizes
%
% INPUTS:
%           
%           spikeTimes:     vector of spike times in samples/frames
%           spikeChannels:  vector where each element is the channel/neuron
%                           number of the corresponding spike in spikeTimes
%           lag:            synchronicity window (s)
%           fs:             sampling frequency / framerate
%           parallel:       if 1, use parallel computing toolbox
%           Time:           Vector of start and end time to calculate STTC on in s 
%           

function adjM = sttc_times_fcn(spikeTimes,spikeChannels,lag,fs,parallel,Time)

numChannel = length(unique(spikeChannels));
chan_labels = unique(spikeChannels);
combChannel = nchoosek(1:numChannel, 2);
A = zeros(1, length(combChannel));
adjM = NaN(numChannel, numChannel);

if parallel == 1
    parfor i = 1:length(combChannel)
        dtv = lag; % [s]
        % spike_times_1 = double(spikeTimes{combChannel(i,1)}.bior1p5/fs);
        % spike_times_2 = double(spikeTimes{combChannel(i,2)}.bior1p5/fs);
        spike_times_1 = find(spikeTimes(:,combChannel(i,1)) == 1) ./ fs;
        spike_times_2 = find(spikeTimes(:,combChannel(i,2)) == 1) ./ fs;
        N1v = int64(length(spike_times_1));
        N2v = int64(length(spike_times_2));
        dtv = double(dtv);
        % Time = double([0 spikeDetectionResult.params.duration]);
        Time = [ 0    size(spikeMatrix,1)/fs ];
        % Time = [ 0    max(spikeTimes)/fs ]; 
        tileCoef = sttc_jeremi(N1v, N2v, dtv, Time, spike_times_1, spike_times_2);
        row = combChannel(i,1);
        col = combChannel(i,2);
        A(i) = tileCoef; % Faster to only get upper triangle so might as well store as vector
    end
else
    for i = 1:length(combChannel)
        dtv = lag; % [s]
        % spike_times_1 = double(spikeTimes{combChannel(i,1)}.bior1p5/fs);
        % spike_times_2 = double(spikeTimes{combChannel(i,2)}.bior1p5/fs);
%         spike_times_1 = find(spikeMatrix(:,combChannel(i,1)) == 1) ./ fs;
%         spike_times_2 = find(spikeMatrix(:,combChannel(i,2)) == 1) ./ fs;
        spike_times_1 = spikeTimes(find(spikeChannels==chan_labels(combChannel(i,1)))) ./ fs;
        spike_times_2 = spikeTimes(find(spikeChannels==chan_labels(combChannel(i,2)))) ./ fs;
        N1v = length(spike_times_1);
        N2v = length(spike_times_2);
        dtv = double(dtv);
        % Time = double([0 spikeDetectionResult.params.duration]);
%         tileCoef = sttc_jeremi(N1v, N2v, dtv, Time, spike_times_1, spike_times_2);
        tileCoef = sttc(N1v, N2v, dtv, Time, spike_times_1, spike_times_2);
        row = combChannel(i,1);
        col = combChannel(i,2);
        A(i) = tileCoef; % Faster to only get upper triangle so might as well store as vector
    end
end

% Vector -> matrix
for i = 1:length(combChannel)
    row = combChannel(i,1);
    col = combChannel(i,2);
    adjM(row, col) = A(i);
    adjM(col, row) = A(i);
end


end